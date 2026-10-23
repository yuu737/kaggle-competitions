"""Observation-authoritative controller; memory stores intent, not simulated assets."""

from copy import deepcopy
from dataclasses import asdict

from .execution import Policy, execute
from .farm_plan import FarmPlan, choose_plan
from .physics import advance
from .planner import choose_project
from .projects import Project, can_install, installed, pending_projects, reconcile_projects
from .rules import CROPS
from .state import State
from .strategy import investment_due, phase


class Controller:
    def __init__(self, policy=None):
        self.policy = policy or Policy()
        self.reset()

    def reset(self):
        self.player = None
        self.last_step = -1
        self.last_plan_day = -1
        self.last_plan_step = -1
        self.project = None
        self.expected = None
        self.history = []
        self.last_trace = {}
        self.farm_plan = FarmPlan()
        self.rotation_after = {}
        self.observed_plants = {}
        self.last_shops = ()
        self.stats = dict(
            calls=0,
            searches=0,
            investments=0,
            predicted_noops=0,
            predicted_discarded=0,
            cash_prediction_mismatches=0,
            private_prediction_mismatches=0,
            project_invalidations=0,
            maximum_search_s=0.0,
            compound_investment_searches=0,
            farm_plan_searches=0,
            farm_plan_adoptions=0,
            farm_projects_installed=0,
            rotation_contracts=0,
        )

    def decide(self, observation, configuration=None):
        state = State.from_observation(observation, configuration)
        if state.player != self.player or state.step <= self.last_step:
            self.reset()
            self.player = state.player
        if self.expected is not None and self.expected.step == state.step:
            self.stats["cash_prediction_mismatches"] += int(
                self.expected.farm["money"] != state.farm["money"]
            )
            self.stats["private_prediction_mismatches"] += int(
                self.expected.private != state.private
            )
        self.last_step = state.step
        self.stats["calls"] += 1
        if self.policy.farm_planning:
            return self._farm_action(state)
        if self.project is not None:
            if installed(state, self.project):
                self.project = None
            else:
                if not can_install(state, self.project):
                    self.project = None
                    self.stats["project_invalidations"] += 1
        pending = pending_projects(state)
        if self.project is not None:
            # One observed seed/animal belongs to the committed intent even if
            # a weed makes the inferred nearest site change. Do not interpret
            # the same purchased item as a second investment elsewhere.
            reserved = False
            other_projects = []
            for project in pending:
                if not reserved and project.item == self.project.item:
                    reserved = True
                    continue
                if project.location != self.project.location:
                    other_projects.append(project)
            pending = [self.project, *other_projects]
        # Acquired inputs are completed before considering another investment.
        # Strategy controls capital redeployment, never creates simulated stock.
        if not pending and investment_due(
            state, self.policy, self.last_plan_day, self.last_plan_step
        ):
            self.last_plan_day = state.day
            self.last_plan_step = state.step
            project, decision = choose_project(state, self.policy)
            self.stats["searches"] += 1
            self.stats["compound_investment_searches"] += int(
                self.policy.investment_strategy == "compound"
            )
            self.stats["maximum_search_s"] = max(
                self.stats["maximum_search_s"], decision["search_seconds"]
            )
            self.history.append(
                dict(
                    step=state.step,
                    strategy=self.policy.investment_strategy,
                    phase=phase(state),
                    **decision,
                )
            )
            self.history = self.history[-128:]
            if project is not None:
                self.project = project
                pending = [project]
                self.stats["investments"] += 1
        action, trace = execute(state, pending, self.policy)
        branch = state.clone()
        prediction = advance(branch, action)
        self.stats["predicted_noops"] += sum(
            r["effect"] is None and r["action"][0] != "PASS" for r in prediction["workers"]
        )
        self.stats["predicted_discarded"] += prediction["discarded"]
        self.expected = branch
        self.last_trace = dict(
            step=state.step, action=deepcopy(action), prediction=prediction, **trace
        )
        return action

    def _farm_action(self, state):
        """Reconcile observations before each decision; retain no predicted assets."""
        plants = {pos: tile["crop"] for pos, tile in state.plants()}
        for pos, item in self.observed_plants.items():
            if plants.get(pos) != item and not CROPS[item]["interval"]:
                # Harvest during day d permits the next planting from day d+1.
                # At an observed day boundary the preceding action was day d-1.
                self.rotation_after[pos] = state.day + int(state.hour != 0)
        self.observed_plants = plants
        rotations = dict(self.farm_plan.rotations)
        for pos, item in plants.items():
            if not CROPS[item]["interval"]:
                rotations.setdefault(pos, item)
        completed = sum(installed(state, p) for p in self.farm_plan.projects)
        self.stats["farm_projects_installed"] += completed
        pending = []
        for project in self.farm_plan.projects:
            if installed(state, project):
                continue
            if can_install(state, project):
                pending.append(project)
            else:
                self.stats["project_invalidations"] += 1
        for pos, item in sorted(rotations.items()):
            project = Project(item, pos)
            if (
                pos not in {p.location for p in pending}
                and state.day >= self.rotation_after.get(pos, state.day)
                and can_install(state, project)
            ):
                pending.append(project)
                self.stats["rotation_contracts"] += 1
        blocked = {pos for pos, day in self.rotation_after.items() if state.day < day}
        pending = reconcile_projects(state, pending, blocked)
        current = FarmPlan(
            self.farm_plan.family,
            pending,
            tuple(sorted(rotations.items())),
            tuple(sorted(self.rotation_after.items())),
            self.farm_plan.service,
        )
        shops = tuple(state.shops)
        due = (
            state.day != self.last_plan_day
            or shops != self.last_shops
            or not pending
            or completed > 0
        )
        final_day = state.settings.last_action_step // state.settings.turns_per_day
        if (
            self.policy.investments
            and state.day < final_day
            and due
            and (self.last_plan_step < 0 or state.step - self.last_plan_step >= 6)
        ):
            self.last_plan_day, self.last_plan_step = state.day, state.step
            self.last_shops = shops
            selected, decision = choose_plan(state, current, self.policy)
            self.stats["searches"] += 1
            self.stats["farm_plan_searches"] += 1
            self.stats["maximum_search_s"] = max(
                self.stats["maximum_search_s"], decision["search_seconds"]
            )
            self.history.append(
                dict(step=state.step, strategy="farm", phase=phase(state), **decision)
            )
            self.history = self.history[-128:]
            if selected is not None:
                current = selected
                self.stats["farm_plan_adoptions"] += 1
                self.stats["investments"] += 1
        self.farm_plan = current
        action, trace = execute(state, current.projects, self.policy)
        branch = state.clone()
        prediction = advance(branch, action)
        self.stats["predicted_noops"] += sum(
            r["effect"] is None and r["action"][0] != "PASS" for r in prediction["workers"]
        )
        self.stats["predicted_discarded"] += prediction["discarded"]
        self.expected = branch
        self.last_trace = dict(
            step=state.step,
            action=deepcopy(action),
            prediction=prediction,
            farm_plan=asdict(current),
            **trace,
        )
        return action

    def diagnostics(self):
        return dict(
            stats=dict(self.stats),
            policy=asdict(self.policy),
            decisions=deepcopy(self.history),
            latest=deepcopy(self.last_trace),
        )
