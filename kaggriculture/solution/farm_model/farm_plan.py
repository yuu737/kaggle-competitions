"""Bounded whole-farm alternatives, valued against the current executable plan.

Plans contain intentions only. Purchased inputs, installed assets, and cash are
always taken from the observation, including in every independent rollout.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
from time import perf_counter

from .execution import execute
from .forecast import SCENARIOS, _daily_outputs, forecast_plan
from .market import price, quote_purchase, quote_sale
from .physics import advance
from .projects import Project, available_sites, can_install, installed, reconcile_projects
from .rules import ANIMALS, CROPS, new_animal, new_plant


@dataclass(frozen=True)
class FarmPlan:
    family: str = "continue"
    projects: tuple = ()
    rotations: tuple = ()
    rotation_after: tuple = ()
    service: tuple = ()


def downside_score(gains):
    """Growth in the central case, with a penalty for its downside uncertainty."""
    central = gains[1]
    return central - 0.5 * max(0.0, central - min(gains))


def binding_projects(state, projects, blocked_sites=()):
    """Preserve real materials and built structures; unfunded future plans may change."""
    from .projects import input_counts

    counts = input_counts(state)
    result = []
    for project in projects:
        if not can_install(state, project):
            continue
        x, y = project.location
        tile = state.farm["tiles"][y][x]
        built = (
            project.animal
            and isinstance(tile, dict)
            and (tile.get("kind") == ANIMALS[project.item]["structure"])
        )
        if counts[project.item] > 0:
            counts[project.item] -= 1
            result.append(project)
        elif built:
            result.append(project)
    return reconcile_projects(state, result, blocked_sites)


def _item_value(state, item):
    """Cheap proposal ordering; full cash/capacity forecasts decide adoption."""
    cfg = state.settings
    end_day = cfg.last_action_step // cfg.turns_per_day
    tile = (
        new_animal(item, state.day)
        if item in ANIMALS
        else new_plant(item, state.day, cfg.turns_per_day)
    )
    assets = [((0, 0), tile)]
    value = 0.0
    actions = 1.0
    capital = ANIMALS[item]["cost"] if item in ANIMALS else CROPS[item]["seed"]
    for day in range(state.day + 1, end_day + 1):
        assets, output, feed = _daily_outputs(assets, day, cfg, 0.95)
        value += sum(
            n * price(product, state.market[product], state.params) for product, n in output.items()
        )
        value -= feed * price("WHEAT", state.market["WHEAT"], state.params)
        actions += (4 if item in ANIMALS else 1.5) + sum(output.values()) / 6
        if not assets and item in CROPS and not CROPS[item]["interval"]:
            # Annual cycles start on the next day, as in the live controller.
            assets = [((0, 0), new_plant(item, day + 1, cfg.turns_per_day))]
            capital += CROPS[item]["seed"]
    return (value - capital) / actions


def candidates(state, current, policy):
    """Compare all eight products before selecting the two physical branches."""
    allowed = policy.allowed_items or (*CROPS, *ANIMALS)
    ranks = sorted(allowed, key=lambda item: (-_item_value(state, item), item))
    annual = [item for item in ranks if item in CROPS and not CROPS[item]["interval"]]
    animals = [item for item in ranks if item in ANIMALS]
    crop = [item for item in ranks if item in CROPS]
    mixed = tuple(([crop[0]] if crop else []) + ([animals[0]] if animals else []))
    bridge_species = ()
    bridge_future = None
    if getattr(policy, "cash_bridge", False) and annual and animals:
        fast_scores = []
        for item in annual:
            spec = CROPS[item]
            if spec["first"] > 2:
                continue
            peak = max(spec["first"], spec["peak"])
            assets = [((0, 0), new_plant(item, state.day, state.settings.turns_per_day))]
            quantity = 0.0
            for day in range(state.day + 1, state.day + peak + 1):
                assets, output, _ = _daily_outputs(assets, day, state.settings)
                quantity += output[item]
            sale, _ = quote_sale(item, int(quantity), state.market[item], state.params)
            score = (sale - spec["seed"]) / (spec["seed"] * peak)
            fast_scores.append((-score, item))
        if fast_scores:
            bridge_species = (min(fast_scores)[1], animals[0])
            bridge_future = annual[0]
    large_batches = bool(getattr(policy, "large_batches", False))
    preserve_rotations = bool(getattr(policy, "preserve_rotations", False))
    # Compare quantities across repeated observed decisions, rather than using
    # two raw or physical slots for the same production family. Including day
    # makes consecutive mornings alternate too (24 turns / 6 is always even).
    large_quantity = large_batches and (state.day + state.step // 6) % 2 == 1
    crop_size = 16 if large_quantity else 8
    animal_size = 8 if large_quantity else 4
    # Every production family survives cheap ordering. Prices today do not
    # capture future shop demand, early cash recycling, or rival saturation.
    families = (
        *(("crop_" + item.lower(), (item,), (crop_size,)) for item in crop),
        *(("livestock_" + item.lower(), (item,), (animal_size,)) for item in animals),
        ("mixed", mixed, (crop_size,)),
    )
    if bridge_species:
        families += (("mixed_bridge", bridge_species, (crop_size,)),)
    cooldown = dict(current.rotation_after)
    blocked = {pos for pos, day in cooldown.items() if state.day < day}
    binding = binding_projects(state, current.projects, blocked)
    occupied = len(state.plants()) + len(state.animals())
    contract_limit = 16 if large_batches else 8
    room = max(
        0, min(contract_limit - len(binding), policy.max_assets - occupied - len(binding))
    )
    emitted = set()
    # Observation and binding stay fixed while proposals are generated. Reuse
    # each product's ordered viable sites across its batch and the mixed plan;
    # only the proposal-local set of occupied sites changes.
    site_choices = {}
    bound_sites = {p.location for p in binding}
    additive_base = (
        reconcile_projects(state, current.projects, blocked)
        if getattr(policy, "additive_growth", False) else ()
    )
    additive_binding = (
        binding_projects(state, additive_base, blocked) if additive_base else ()
    )
    for family, species, sizes in families:
        additive = (
            bool(getattr(policy, "additive_growth", False))
            and family.startswith("crop_")
            and species[0] in CROPS
            and bool(CROPS[species[0]]["interval"])
        )
        base = additive_base if additive else binding
        base_sites = {p.location for p in base}
        family_room = (
            max(0, min(contract_limit - len(base), policy.max_assets - occupied - len(base)))
            if additive else room
        )
        if not species or not family_room:
            continue
        for size in sizes:
            projects = list(base)
            used = {p.location for p in projects}
            for index in range(min(family_room, size)):
                item = species[index % len(species)]
                if item not in site_choices:
                    choices = (
                        Project(item, pos, land)
                        for _, pos, land in sorted(
                            available_sites(state, item, include_next_land=True),
                            key=lambda site: (site[2], site[0], site[1]),
                        )
                        if pos not in bound_sites and pos not in blocked
                    )
                    site_choices[item] = tuple(p for p in choices if can_install(state, p))
                project = next(
                    (p for p in site_choices[item] if p.location not in used), None
                )
                if project is None:
                    break
                projects.append(project)
                used.add(project.location)
            if additive:
                protected = {p.location for p in additive_binding}
                projects = (
                    list(additive_binding)
                    + projects[len(base):]
                    + [p for p in base if p.location not in protected]
                )
            real_or_bound = (
                {pos for pos, _ in state.plants()} | base_sites | set(cooldown)
            )
            rotations = {pos: item for pos, item in current.rotations if pos in real_or_bound}
            replacement = (
                species[0] if family.startswith("crop_") and species[0] in annual else None
            )
            for pos, tile in state.plants():
                if not CROPS[tile["crop"]]["interval"]:
                    if family == "mixed_bridge" or preserve_rotations or additive:
                        rotations.setdefault(pos, tile["crop"])
                    else:
                        rotations[pos] = replacement or tile["crop"]
            for project in projects:
                if project.item in CROPS and not CROPS[project.item]["interval"]:
                    if project.location in base_sites and (
                        family == "mixed_bridge" or preserve_rotations or additive
                    ):
                        rotations.setdefault(project.location, project.item)
                    elif family == "mixed_bridge":
                        # The first contract buys the fast crop. Only its
                        # observed harvest and next-day cooldown permit
                        # this future crop through the existing controller.
                        rotations[project.location] = bridge_future
                    else:
                        rotations[project.location] = project.item
            plan = FarmPlan(
                family,
                tuple(projects),
                tuple(sorted(rotations.items())),
                current.rotation_after,
                current.service,
            )
            signature = (plan.projects, plan.rotations)
            if signature not in emitted and signature != (current.projects, current.rotations):
                emitted.add(signature)
                yield plan

    if getattr(policy, "rotation_only", False) and annual:
        live_annual = {
            pos: tile["crop"]
            for pos, tile in state.plants()
            if not CROPS[tile["crop"]]["interval"]
        }
        eligible = set(live_annual)
        current_rotations = dict(current.rotations)
        for pos in set(cooldown) & current_rotations.keys():
            x, y = pos
            tile = state.farm["tiles"][y][x]
            if tile is None or isinstance(tile, dict) and tile.get("kind") == "WEED":
                eligible.add(pos)
        real_or_bound = {pos for pos, _ in state.plants()} | bound_sites | set(cooldown)
        preserved = {
            pos: item for pos, item in current.rotations if pos in real_or_bound
        }
        # A value-per-action single-product proxy misses portfolio liquidity.
        # Put the observed first-sale return on seed cash first; the existing
        # coarse whole-farm forecasts select among inspected alternatives.
        target_scores = []
        if eligible - bound_sites:
            for item in annual:
                spec = CROPS[item]
                assets = [((0, 0), new_plant(item, state.day, state.settings.turns_per_day))]
                sale = 0.0
                remaining = (
                    state.settings.last_action_step // state.settings.turns_per_day - state.day
                )
                horizon = max(1, min(spec["peak"], remaining))
                for wait in range(1, horizon + 1):
                    assets, output, _ = _daily_outputs(assets, state.day + wait, state.settings)
                    if output[item] > 0:
                        sale, _ = quote_sale(
                            item, int(output[item]), state.market[item], state.params
                        )
                        break
                score = (sale - spec["seed"]) / (spec["seed"] * wait)
                target_scores.append((-score, item))
        for _, target in sorted(target_scores):
            rotations = dict(preserved)
            changed = False
            for pos in sorted(eligible - bound_sites):
                previous = rotations.get(pos, live_annual.get(pos))
                changed |= previous != target
                rotations[pos] = target
            plan = FarmPlan(
                "rotation", binding, tuple(sorted(rotations.items())),
                current.rotation_after, current.service,
            )
            signature = (plan.projects, plan.rotations)
            if (
                changed
                and signature not in emitted
                and signature != (current.projects, current.rotations)
            ):
                emitted.add(signature)
                yield plan


def physical_rollout(state, plan, policy, deadline):
    """The live executor for one day; purchased assets are never preinstalled."""
    branch = state.clone()
    projects = tuple(plan.projects)
    rotations = dict(plan.rotations)
    next_day = dict(plan.rotation_after)
    public = [
        (pos, deepcopy(tile))
        for pos, tile in state.tiles(opponent=True)
        if isinstance(tile, dict) and ("animal" in tile or tile.get("kind") == "PLANT")
    ]
    remainder = {item: 0.0 for item in state.market}
    previous_day = None
    noops = discarded = unserved = deaths = 0
    minimum_cash = float(branch.farm["money"])
    animal_turns = care_actions = fertilizer_actions = 0
    service_cohorts, cared, fertilizer_collected = set(), set(), set()
    fertilizer_opportunities, completed_days = set(), set()
    for _ in range(min(policy.rollout_turns, state.settings.last_action_step - state.step + 1)):
        if perf_counter() >= deadline:
            return dict(complete=False, state=branch, reason="physical budget")
        if previous_day != branch.day:
            previous_day = branch.day
            for pos, tile in branch.animals():
                # New placements lack a full daily service opportunity.
                if tile["placed_day"] < branch.day:
                    key = (pos, branch.day)
                    service_cohorts.add(key)
                    if tile.get("cared_today", False):
                        cared.add(key)
            unfed = sum(not tile.get("fed_today", False) for _, tile in public if "animal" in tile)
            public, outputs, _ = _daily_outputs(public, branch.day, branch.settings)
            if branch.day < branch.settings.last_action_step // branch.settings.turns_per_day:
                cost, branch.market["WHEAT"] = quote_purchase(
                    "WHEAT", unfed, branch.market["WHEAT"], branch.params
                )
                branch.opponent["money"] -= cost
        for pos, item in rotations.items():
            x, y = pos
            tile = branch.farm["tiles"][y][x]
            eligible = tile is None or isinstance(tile, dict) and tile.get("kind") == "WEED"
            project = Project(item, pos)
            if (
                eligible
                and branch.day >= next_day.get(pos, state.day)
                and can_install(branch, project)
            ):
                if pos not in {p.location for p in projects}:
                    projects += (project,)
        blocked = {pos for pos, day in next_day.items() if branch.day < day}
        projects = reconcile_projects(branch, projects, blocked)
        before = {pos: tile["crop"] for pos, tile in branch.plants()}
        injected = {}
        for item, n in outputs.items():
            remainder[item] += n / branch.settings.turns_per_day
            injected[item] = int(remainder[item] + 1e-9)
            remainder[item] = max(0.0, remainder[item] - injected[item])
        action, _ = execute(branch, projects, policy)
        before_animals = {pos for pos, _ in branch.animals()}
        worker_positions = list(branch.positions)
        animal_turns += len(before_animals)
        action_day = branch.day
        fertilizer_opportunities.update(
            (pos, branch.day)
            for pos, tile in branch.animals()
            if tile.get("fertilizer_available", False) and (pos, branch.day) in service_cohorts
        )
        result = advance(branch, action)
        care_actions += sum(row["effect"] == "CARE" for row in result["workers"])
        fertilizer_actions += sum(
            row["effect"] == "COLLECT_FERTILIZER" for row in result["workers"]
        )
        for row in result["workers"]:
            key = (worker_positions[row["unit"]], action_day)
            if key not in service_cohorts:
                continue
            if row["effect"] == "CARE":
                cared.add(key)
            elif row["effect"] == "COLLECT_FERTILIZER":
                fertilizer_collected.add(key)
        # Sequential background deliveries are an explicit approximation.
        # Their receipts and public biological state must reach the tail too.
        for item, count in injected.items():
            revenue, branch.market[item] = quote_sale(
                item, count, branch.market[item], branch.params
            )
            branch.opponent["money"] += revenue
        if branch.hour == 0:
            completed_days.add(action_day)
            deaths += len(before_animals - {pos for pos, _ in branch.animals()})
            unserved += sum(tile.get("consecutive_unfed", 0) > 0 for _, tile in branch.animals())
            unserved += sum(tile.get("consecutive_unwatered", 0) > 0 for _, tile in branch.plants())
            harvested = {
                worker_positions[row["unit"]]
                for row in result["workers"]
                if row["effect"] == "HARVEST"
            }
            for pos in before:
                x, y = pos
                tile = branch.farm["tiles"][y][x]
                if isinstance(tile, dict) and tile.get("kind") == "WEED" and pos not in harvested:
                    deaths += 1
            for (x, y), tile in list(branch.tiles(opponent=True)):
                if isinstance(tile, dict) and ("animal" in tile or tile.get("kind") == "PLANT"):
                    branch.opponent["tiles"][y][x] = None
            for (x, y), tile in public:
                branch.opponent["tiles"][y][x] = deepcopy(tile)
        for pos, item in before.items():
            x, y = pos
            tile = branch.farm["tiles"][y][x]
            if not (isinstance(tile, dict) and tile.get("crop") == item):
                next_day[pos] = branch.day + int(branch.hour != 0)
        noops += sum(
            row["effect"] is None and row["action"][0] != "PASS" for row in result["workers"]
        )
        discarded += result["discarded"]
        minimum_cash = min(minimum_cash, branch.farm["money"])
    blocked = {pos for pos, day in next_day.items() if branch.day < day}
    # A rolling 24-turn window ends mid-day in most calls. Measure completed
    # animal-days and credit service already observed earlier that same day;
    # a not-yet-complete next day must not dilute a perfect daily service rate.
    completed_cohorts = {key for key in service_cohorts if key[1] in completed_days}
    completed_fertilizer = {key for key in fertilizer_opportunities if key[1] in completed_days}
    previous_service = dict(plan.service)
    service = (
        (
            "care",
            len(cared & completed_cohorts) / len(completed_cohorts)
            if completed_cohorts
            else previous_service.get("care", 1.0),
        ),
        (
            "fertilizer",
            len(fertilizer_collected & completed_fertilizer) / len(completed_fertilizer)
            if completed_fertilizer
            else previous_service.get("fertilizer", 1.0),
        ),
    )
    tail_plan = FarmPlan(
        plan.family,
        reconcile_projects(branch, projects, blocked),
        plan.rotations,
        tuple(sorted(next_day.items())),
        service,
    )
    return dict(
        complete=True,
        state=branch,
        plan=tail_plan,
        noops=noops,
        discarded=discarded,
        unserved=unserved,
        deaths=deaths,
        minimum_cash=minimum_cash,
        installed=sum(installed(branch, p) for p in plan.projects),
        service_measurement=dict(
            animal_turns=animal_turns,
            care_actions=care_actions,
            fertilizer_actions=fertilizer_actions,
            fertilizer_opportunities=len(completed_fertilizer),
            completed_animal_days=len(completed_cohorts),
            fractions=dict(service),
        ),
    )


def _cases(state, plan, policy, deadline, rival_rotations=None):
    result = []
    for scenario in SCENARIOS:
        if perf_counter() >= deadline:
            return None
        options = {"rival_rotations": rival_rotations} if rival_rotations is not None else {}
        case = forecast_plan(
            state, scenario, plan, policy.max_workers, deadline=deadline, **options
        )
        if case is None:
            return None
        result.append(case)
    return result if perf_counter() < deadline else None


def _reuse_rotation_prefix(state, current, plan, baseline):
    """Reuse a completed day only if changed future crops could not have fired."""
    if (
        not baseline["complete"]
        or baseline["state"].step != state.step + 24
        or plan.projects != current.projects
        or plan.rotation_after != current.rotation_after
        or plan.service != current.service
    ):
        return None
    old, proposed = dict(current.rotations), dict(plan.rotations)
    if old.keys() != proposed.keys():
        return None
    changed = {pos: item for pos, item in proposed.items() if item != old[pos]}
    if not changed:
        return None
    for x, y in changed:
        initial = state.farm["tiles"][y][x]
        end = baseline["state"].farm["tiles"][y][x]
        if (
            not isinstance(initial, dict)
            or initial.get("kind") != "PLANT"
            or initial.get("crop") not in CROPS
            or CROPS[initial["crop"]]["interval"]
            or not isinstance(end, dict)
            or end.get("kind") != "PLANT"
            or end.get("crop") != initial["crop"]
            or end.get("planted_day") != initial.get("planted_day")
            or initial.get("planted_day") is None
        ):
            return None
        life = initial.get("max_lifespan_step", -1)
        if life >= 0 and baseline["state"].step >= life:
            return None
    tail = baseline["plan"]
    rotations = dict(tail.rotations)
    rotations.update(changed)
    result = dict(baseline)
    result["state"] = baseline["state"].clone()
    result["plan"] = FarmPlan(
        plan.family, tail.projects, tuple(sorted(rotations.items())),
        tail.rotation_after, tail.service,
    )
    return result


def choose_plan(state, current, policy):
    began = perf_counter()
    budget = min(0.30, policy.search_seconds)
    deadline = began + budget
    coarse_deadline = began + min(0.08, 0.30 * budget)
    report = dict(
        model="whole-farm staged investment + executable day",
        selected=None,
        raw_candidates=0,
        physical_candidates=0,
        candidates=[],
        rejected=[],
        coarse_budget_exhausted=False,
    )

    def finish(selected=None):
        report["search_seconds"] = perf_counter() - began
        report["selected"] = asdict(selected) if selected else None
        return selected, report

    if (
        not policy.investments
        or state.day >= state.settings.last_action_step // state.settings.turns_per_day
    ):
        return finish()
    rival_options = {}
    if getattr(policy, "rival_cycles", False):
        # Preserve only footprints visible at the real decision. A crop removed
        # in the one-day prefix remains a future, cash-funded rival hypothesis.
        rival_options["rival_rotations"] = tuple(
            (pos, tile["crop"], state.day + 1)
            for pos, tile in state.tiles(opponent=True)
            if isinstance(tile, dict)
            and tile.get("kind") == "PLANT"
            and not CROPS[tile["crop"]]["interval"]
        )
    central = forecast_plan(
        state, SCENARIOS[1], current, policy.max_workers, deadline=coarse_deadline,
        **rival_options,
    )
    if central is None:
        report["coarse_budget_exhausted"] = True
        return finish()
    ranked = []
    proposals = list(candidates(state, current, policy))
    # Preserve time for the actual executor and all three tail checks. Rotate
    # the starting family using observed time so an expensive coarse pass does
    # not systematically hide the same products on every decision.
    if proposals:
        tick = state.step // 6
        crops = [p for p in proposals if p.family.startswith("crop_")]
        animals = [p for p in proposals if p.family.startswith("livestock_")]
        mixtures = [p for p in proposals if p.family.startswith("mixed")]
        rotations = [p for p in proposals if p.family == "rotation"]
        representatives = []
        # A bounded scan must compare economic designs, rather than spending
        # the whole initial allowance on products from one design. Rotate the
        # crop and animal representatives using only the observed turn.
        for group in (crops, animals, mixtures):
            if group:
                representatives.append(group[tick % len(group)])
        if getattr(policy, "rotation_only", False) and rotations:
            representatives.append(rotations[0])
        elif getattr(policy, "cash_bridge", False):
            # The remaining raw slot compares the financing policy too; cheap
            # crop ordering must not hide the alternative at the first turn.
            representatives.extend(p for p in mixtures if p not in representatives)
        offset = 3 * tick % len(proposals)
        remainder = proposals[offset:] + proposals[:offset]
        proposals = representatives + [p for p in remainder if p not in representatives]
    for plan in proposals:
        if report["raw_candidates"] >= policy.candidate_limit:
            break
        if perf_counter() >= coarse_deadline:
            report["coarse_budget_exhausted"] = True
            break
        report["raw_candidates"] += 1
        case = forecast_plan(
            state, SCENARIOS[1], plan, policy.max_workers, deadline=coarse_deadline,
            **rival_options,
        )
        if case is None:
            report["coarse_budget_exhausted"] = True
            break
        gain = case.margin - central.margin
        if case.feasible and gain > policy.minimum_gain:
            ranked.append((gain, plan))
        else:
            report["rejected"].append(
                dict(family=plan.family, gain=gain, reason="central cash/capacity or gain")
            )
    ranked.sort(key=lambda row: -row[0])
    if not ranked or perf_counter() >= deadline:
        return finish()
    base = physical_rollout(state, current, policy, deadline)
    if not base["complete"]:
        report["rejected"].append(dict(reason="baseline physical budget"))
        return finish()
    base_cases = _cases(base["state"], base["plan"], policy, deadline, **rival_options)
    if base_cases is None:
        report["rejected"].append(dict(reason="baseline tail forecast budget"))
        return finish()
    report["baseline"] = dict(
        margins=[c.margin for c in base_cases],
        cash=base["state"].farm["money"],
        service=dict(base["plan"].service),
    )
    best, best_score = None, policy.minimum_gain
    # Two sizes of the same policy are not two different strategic alternatives.
    # Keep the best proposal per family before spending the physical budget.
    distinct = []
    seen_families = set()
    for coarse_gain, plan in ranked:
        if plan.family not in seen_families:
            seen_families.add(plan.family)
            distinct.append((coarse_gain, plan))
    # A larger livestock proposal must not occupy both branch slots forever.
    # Reserve one available crop-policy branch and one other branch. Alternate
    # their order by observed day so a hard budget cannot always hide crops.
    crop_branches = [
        row
        for row in distinct
        if row[1].family.startswith("crop_") or row[1].family in ("rotation", "perennial")
    ]
    other_branches = [row for row in distinct if row not in crop_branches]
    if policy.physical_candidates == 1:
        # A single executable branch must be the economic front-runner. A
        # diversity preference cannot replace it with a lower-ranked family.
        # This saves a full duplicate day of execution, leaving time to finish
        # the incumbent comparison and all three scenarios for the best idea.
        chosen = distinct
    elif crop_branches and other_branches:
        chosen = [other_branches[0], crop_branches[0]]
        if state.day % 2:
            chosen.reverse()
    else:
        chosen = distinct
    for coarse_gain, plan in chosen[: policy.physical_candidates]:
        if perf_counter() >= deadline:
            break
        report["physical_candidates"] += 1
        trial = None
        if getattr(policy, "rotation_only", False) and plan.family == "rotation":
            trial = _reuse_rotation_prefix(state, current, plan, base)
        prefix_reused = trial is not None
        if trial is None:
            trial = physical_rollout(state, plan, policy, deadline)
        if (
            not trial["complete"]
            or trial["noops"]
            or trial["discarded"]
            or trial["unserved"] > base["unserved"]
            or trial["deaths"] > base["deaths"]
        ):
            problems = []
            if not trial["complete"]:
                problems.append("physical budget")
            if trial.get("noops", 0):
                problems.append("noop")
            if trial.get("discarded", 0):
                problems.append("overflow")
            if trial.get("unserved", 0) > base["unserved"]:
                problems.append("maintenance loss")
            if trial.get("deaths", 0) > base["deaths"]:
                problems.append("asset loss")
            report["rejected"].append(
                dict(
                    family=plan.family,
                    reason="physical budget/noop/overflow/maintenance loss",
                    details=problems,
                )
            )
            continue
        from .projects import input_counts

        new_contracts = [p for p in plan.projects if p not in current.projects]
        start_inputs, end_inputs = input_counts(state), input_counts(trial["state"])
        progress = any(installed(trial["state"], p) for p in new_contracts) or any(
            end_inputs[item] > start_inputs[item] for item in end_inputs
        )
        if new_contracts and not progress:
            report["rejected"].append(
                dict(family=plan.family, reason="no productive prefix progress")
            )
            continue
        tail = _cases(trial["state"], trial["plan"], policy, deadline, **rival_options)
        if tail is None:
            report["rejected"].append(dict(family=plan.family, reason="tail forecast budget"))
            break
        gains = [c.margin - b.margin for c, b in zip(tail, base_cases, strict=True)]
        score = downside_score(gains)
        feasible = all(c.feasible for c in tail)
        report["candidates"].append(
            dict(
                family=plan.family,
                projects=len(plan.projects),
                coarse_gain=coarse_gain,
                gains=gains,
                score=score,
                feasible=feasible,
                installed_first_day=trial["installed"],
                minimum_prefix_cash=trial["minimum_cash"],
                service=dict(trial["plan"].service),
                **(
                    {"prefix_reused": prefix_reused}
                    if getattr(policy, "rotation_only", False) else {}
                ),
            )
        )
        if feasible and gains[1] > 0 and score > best_score:
            best = FarmPlan(
                plan.family,
                plan.projects,
                plan.rotations,
                plan.rotation_after,
                trial["plan"].service,
            )
            best_score = score
    return finish(best)
