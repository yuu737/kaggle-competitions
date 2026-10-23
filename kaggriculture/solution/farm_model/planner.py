"""Incremental investment search with a shared executor and bounded branches."""

from copy import deepcopy
from dataclasses import asdict
from time import perf_counter

from .execution import execute
from .forecast import SCENARIOS, _daily_outputs, forecast, proposals, running_reserve
from .physics import advance
from .projects import installed
from .rules import ANIMALS, CROPS, LAND_COSTS


def _forecast_cases(state, policy, deadline, project=None):
    values = []
    for scenario in SCENARIOS:
        if perf_counter() >= deadline:
            return None
        result = forecast(state, scenario, project, policy.max_workers)
        values.append(result)
    return values if perf_counter() < deadline else None


def physical_rollout(state, project, policy, deadline):
    branch = state.clone()
    steps = min(policy.rollout_turns, state.settings.last_action_step - state.step + 1)
    projects = () if project is None else (project,)
    public = [
        (pos, deepcopy(tile))
        for pos, tile in state.tiles(opponent=True)
        if isinstance(tile, dict) and ("animal" in tile or tile.get("kind") == "PLANT")
    ]
    remainder = {item: 0.0 for item in branch.market}
    outputs = None
    previous_day = None
    discarded = 0
    noops = 0
    for _ in range(steps):
        if perf_counter() >= deadline:
            return dict(complete=False, reason="search budget", state=branch)
        if previous_day != branch.day:
            previous_day = branch.day
            public, outputs, _ = _daily_outputs(public, branch.day, branch.settings)
        injected = {}
        for item, n in outputs.items():
            remainder[item] += n / branch.settings.turns_per_day
            injected[item] = int(remainder[item])
            remainder[item] -= injected[item]
        action, _ = execute(branch, projects, policy)
        result = advance(branch, action, external_sales=injected)
        discarded += result["discarded"]
        noops += sum(r["effect"] is None and r["action"][0] != "PASS" for r in result["workers"])
    return dict(
        complete=True,
        state=branch,
        discarded=discarded,
        noops=noops,
        installed=project is None or installed(branch, project),
        steps=steps,
    )


def choose_project(state, policy):
    began = perf_counter()
    deadline = began + policy.search_seconds
    report = dict(
        selected=None,
        raw_candidates=0,
        physical_candidates=0,
        rejected=[],
        candidates=[],
        baseline={},
        model="basic v1 daily cash + shared-executor rollout",
    )
    if not policy.investments or state.step >= state.settings.last_action_step:
        report["search_seconds"] = perf_counter() - began
        return None, report
    base_cases = _forecast_cases(state, policy, deadline)
    if base_cases is None:
        report["rejected"].append(dict(reason="baseline forecast exceeded budget"))
        report["search_seconds"] = perf_counter() - began
        return None, report
    candidates = []
    for project in proposals(state, policy.allowed_items, policy.max_assets)[
        : policy.candidate_limit
    ]:
        if perf_counter() >= deadline:
            break
        report["raw_candidates"] += 1
        capital = ANIMALS[project.item]["cost"] if project.animal else CROPS[project.item]["seed"]
        if project.needs_land:
            capital += LAND_COSTS[len(state.farm["unlocked_quadrants"]) - 1]
        if state.farm["money"] < capital + running_reserve(state, int(project.animal)):
            report["rejected"].append(dict(project=asdict(project), reason="capital/runway"))
            continue
        cases = _forecast_cases(state, policy, deadline, project)
        if cases is None:
            report["rejected"].append(
                dict(project=asdict(project), reason="coarse forecast budget")
            )
            break
        gains = [c.terminal_cash - b.terminal_cash for c, b in zip(cases, base_cases, strict=True)]
        if not all(c.feasible for c in cases) or min(gains) <= policy.minimum_gain:
            report["rejected"].append(
                dict(
                    project=asdict(project),
                    gains=gains,
                    reason="nonpositive robust gain or capacity/cash",
                )
            )
            continue
        candidates.append((min(gains) + 0.25 * sum(gains) / len(gains), project, gains))
    candidates.sort(key=lambda row: (-row[0], row[1].item, row[1].location))
    if not candidates:
        report["search_seconds"] = perf_counter() - began
        return None, report
    baseline = physical_rollout(state, None, policy, deadline)
    if not baseline["complete"]:
        report["rejected"].append(dict(reason="baseline rollout budget"))
        report["search_seconds"] = perf_counter() - began
        return None, report
    base_tail = _forecast_cases(baseline["state"], policy, deadline)
    if base_tail is None:
        report["rejected"].append(dict(reason="baseline tail budget"))
        report["search_seconds"] = perf_counter() - began
        return None, report
    report["baseline"] = dict(
        terminal_cash=[b.terminal_cash for b in base_tail],
        rollout_cash=baseline["state"].farm["money"],
        rollout_noops=baseline["noops"],
        rollout_discarded=baseline["discarded"],
    )
    selected = None
    best_score = 0.0
    for _, project, coarse_gains in candidates[: policy.physical_candidates]:
        if perf_counter() >= deadline:
            break
        report["physical_candidates"] += 1
        trial = physical_rollout(state, project, policy, deadline)
        if (
            not trial["complete"]
            or not trial.get("installed")
            or trial.get("noops")
            or trial.get("discarded")
        ):
            report["rejected"].append(
                dict(
                    project=asdict(project),
                    reason="physical rollout incomplete/unfinished/noop/overflow",
                )
            )
            continue
        tail = _forecast_cases(trial["state"], policy, deadline)
        if tail is None:
            report["rejected"].append(dict(project=asdict(project), reason="candidate tail budget"))
            break
        gains = [c.terminal_cash - b.terminal_cash for c, b in zip(tail, base_tail, strict=True)]
        score = min(gains) + 0.25 * sum(gains) / len(gains)
        safe = all(c.feasible for c in tail) and min(gains) > policy.minimum_gain
        report["candidates"].append(
            dict(
                project=asdict(project),
                coarse_gains=coarse_gains,
                realized_rollout_cash=trial["state"].farm["money"],
                gains=gains,
                score=score,
                feasible=safe,
                costs=[
                    dict(feed=c.feed_cost, wages=c.wages, minimum_cash=c.minimum_cash) for c in tail
                ],
            )
        )
        if safe and score > best_score:
            selected, best_score = project, score
    report["selected"] = asdict(selected) if selected else None
    report["search_seconds"] = perf_counter() - began
    return selected, report
