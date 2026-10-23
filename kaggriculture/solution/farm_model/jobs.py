"""Explicit work contracts and bounded, resource-aware worker assignment."""

from dataclasses import dataclass, field

from .market import price
from .physics import apply_unit
from .projects import can_install, install_deadline, installed
from .rules import ANIMALS, CROPS, PRODUCTS
from .state import distance, move_toward, nearest_shed


@dataclass(frozen=True)
class Job:
    identity: str
    location: tuple
    action: tuple
    deadline: int
    value: float
    requires: tuple = ()
    mandatory: bool = False
    worker: int | None = None
    reason: str = ""
    first_feed: bool = False
    batch_feed: bool = False


@dataclass
class Assignment:
    actions: list
    selected: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    shadow: object = None


def _sell_value(state, product, count=1):
    return count * price(product, state.market[product], state.params)


def _production_after_care(state, tile):
    spec = ANIMALS[tile["animal"]]
    first = tile["placed_day"] + spec["first"]
    day = max(state.day + 2, first)
    return day + (first - day) % spec["interval"]


def generate_jobs(state, projects=(), prepared_livestock=False):
    jobs = []
    cfg = state.settings
    day_deadline = state.end_of_day_step
    last = cfg.last_action_step
    final_day = last // cfg.turns_per_day
    for (x, y), tile in state.tiles():
        if not isinstance(tile, dict):
            continue
        pos = (x, y)
        delivery = distance(pos, nearest_shed(pos, cfg)) + 1
        harvest_deadline = last - delivery  # harvest, travel, DROP then market
        if "animal" in tile:
            spec = ANIMALS[tile["animal"]]
            product = spec["product"]
            future = max(0, final_day - state.day)
            if not tile["fed_today"] and future > 0:
                # Includes avoided escape and saleable fertilizer. Value is a
                # scheduling heuristic, not the investment objective.
                urgent = tile["consecutive_unfed"] >= 1
                value = _sell_value(state, product, min(4, future)) + _sell_value(
                    state, "FERTILIZER"
                )
                jobs.append(
                    Job(
                        f"feed:{x}:{y}",
                        pos,
                        ("FEED",),
                        day_deadline,
                        value,
                        (("WHEAT", 1, "carry"),),
                        True,
                        reason="escape risk" if urgent else "daily feed",
                        first_feed=prepared_livestock and tile["placed_day"] == state.day,
                        batch_feed=prepared_livestock,
                    )
                )
            if tile.get("yield_units", 0) and harvest_deadline >= state.step:
                jobs.append(
                    Job(
                        f"harvest:{x}:{y}",
                        pos,
                        ("HARVEST",),
                        min(day_deadline, harvest_deadline),
                        _sell_value(state, product, tile["yield_units"]),
                        reason="collect produced inventory",
                    )
                )
            if tile.get("fertilizer_available") and harvest_deadline >= state.step:
                jobs.append(
                    Job(
                        f"fertilizer:{x}:{y}",
                        pos,
                        ("COLLECT_FERTILIZER",),
                        min(day_deadline, harvest_deadline),
                        _sell_value(state, "FERTILIZER"),
                        reason="daily nonaccumulating yield",
                    )
                )
            if (
                not tile["cared_today"]
                and _production_after_care(state, tile) <= final_day
                and (tile["fed_today"] or future > 0)
            ):
                jobs.append(
                    Job(
                        f"care:{x}:{y}",
                        pos,
                        ("CARE",),
                        day_deadline,
                        _sell_value(state, product) * 0.8,
                        reason="bonus at a later production event",
                    )
                )
        elif tile.get("kind") == "PLANT":
            spec = CROPS[tile["crop"]]
            age = state.day - tile["planted_day"]
            mature = age >= spec["first"] and tile.get("yield_units", 0) > 0
            remaining = last - state.step
            peak = min(spec["peak"], 10) if tile["crop"] == "MELON" else spec["peak"]
            harvest_now = mature and (
                bool(spec["interval"]) or age >= peak or remaining <= cfg.turns_per_day + delivery
            )
            # Water before harvesting when it earns a bonus this day.
            bonus_water = (
                not spec["interval"]
                and (spec["peak"] + 1) // 2 <= age <= spec["peak"]
                and tile["yield_units"] < spec["cap"]
            )
            terminal = state.day == final_day
            water_deadline = harvest_deadline - 1 if terminal else day_deadline
            water_profitable = not terminal or (mature and bonus_water)
            if (
                not tile["watered_today"]
                and (not harvest_now or bonus_water or (spec["interval"] and not terminal))
                and water_profitable
                and water_deadline >= state.step
            ):
                jobs.append(
                    Job(
                        f"water:{x}:{y}",
                        pos,
                        ("WATER",),
                        water_deadline,
                        _sell_value(
                            state, tile["crop"], 1 if terminal else max(1, tile["yield_units"])
                        ),
                        mandatory=not terminal,
                        reason="cashable final yield bonus"
                        if terminal
                        else "same-day survival or yield bonus",
                    )
                )
            if harvest_now and (tile["watered_today"] or not bonus_water or terminal):
                life = tile.get("max_lifespan_step", -1)
                deadline = min(harvest_deadline, day_deadline)
                if life >= 0:
                    # Lifespan starts alternate-turn decay, not a harvest ban.
                    # Workers act before decay, including its final positive turn.
                    next_decay = max(life, state.step)
                    next_decay += (next_decay - life) % 2
                    deadline = min(deadline, next_decay + 2 * (tile["yield_units"] - 1))
                if deadline >= state.step:
                    jobs.append(
                        Job(
                            f"harvest:{x}:{y}",
                            pos,
                            ("HARVEST",),
                            deadline,
                            _sell_value(state, tile["crop"], tile["yield_units"]),
                            reason="maturity/decay/final liquidation",
                        )
                    )

    for project in projects:
        if installed(state, project) or not can_install(state, project):
            continue
        x, y = project.location
        tile = state.farm["tiles"][y][x]
        if tile == "LOCKED":
            continue
        identity = f"project:{project.item}:{x}:{y}"
        project_deadline = min(day_deadline, install_deadline(state, project))
        if isinstance(tile, dict) and tile.get("kind") == "WEED":
            jobs.append(
                Job(
                    identity,
                    (x, y),
                    ("DIG",),
                    project_deadline - (2 if project.animal else 1),
                    80.0,
                    reason="prepare committed project",
                )
            )
        elif project.animal:
            spec = ANIMALS[project.item]
            if tile is None:
                jobs.append(
                    Job(
                        identity,
                        (x, y),
                        ("BUILD_" + spec["structure"],),
                        project_deadline - 1,
                        100.0,
                        reason="prepare livestock structure",
                    )
                )
            elif (
                isinstance(tile, dict)
                and tile.get("kind") == spec["structure"]
                and "animal" not in tile
            ):
                needs_first_feed = prepared_livestock and state.day < final_day
                requires = ((project.item, 1, "carry"),)
                if needs_first_feed:
                    requires += (("WHEAT", 1, "carry"),)
                jobs.append(
                    Job(
                        identity,
                        (x, y),
                        ("PLACE", project.item),
                        min(project_deadline, day_deadline - int(needs_first_feed)),
                        200.0,
                        requires,
                        reason="complete purchased livestock investment",
                    )
                )
        elif tile is None and state.hour < cfg.turns_per_day - 1:
            jobs.append(
                Job(
                    identity,
                    (x, y),
                    ("PLANT", project.item),
                    min(project_deadline, day_deadline - 1),
                    120.0,
                    ((project.item, 1, "seed"),),
                    reason="plant only when same-day watering can follow",
                )
            )

    unfed = (
        sum(not tile["fed_today"] for _, tile in state.animals()) if state.day < final_day else 0
    )
    for unit, (pos, bag) in enumerate(
        zip(state.positions, state.private["inventories"], strict=True)
    ):
        goods = {
            item: n
            for item, n in bag.items()
            if item in PRODUCTS and n > 0 and (item != "WHEAT" or unfed == 0)
        }
        if not goods:
            continue
        access = nearest_shed(pos, cfg)
        travel = distance(pos, access)
        if state.step + travel > last:
            continue
        total = sum(goods.values())
        has_input = any(n > 0 and item not in goods for item, n in bag.items())
        room = cfg.shed_capacity - state.shed_used()
        if not has_input and total <= room:
            action = ("DROP",)
        else:
            item = max(goods, key=lambda p: _sell_value(state, p, goods[p]))
            count = min(goods[item], room)
            if count <= 0:
                continue  # market will free space after this turn's workers
            action = ("PLACE", item, count)
        # Auto-drop normally handles distant cargo at day end. Final-day cargo
        # must physically reach the shed before the final market phase.
        urgency = 2.0 if state.day == final_day else (0.8 if state.hour >= 16 else 0.3)
        jobs.append(
            Job(
                f"deliver:{unit}",
                access,
                action,
                last if state.day == final_day else day_deadline,
                sum(_sell_value(state, p, n) for p, n in goods.items()) * urgency,
                worker=unit,
                reason="physical delivery before sale",
            )
        )
    return jobs


def route(job, unit, state):
    if job.worker is not None and job.worker != unit:
        return None
    # Routing one worker does not need to rebuild every worker's position.
    if unit == 0:
        pos = tuple(state.farm["farmer"])
    elif unit > 0:
        pos = tuple(state.farm["hands"][unit - 1])
    else:
        pos = state.positions[unit]
    bag = state.private["inventories"][unit]
    for item, n, source in job.requires:
        if source == "seed":
            if state.private["seeds"].get(item, 0) < n:
                return None
        elif bag.get(item, 0) < n:
            available = state.private["shed"].get(item, 0)
            if available < n:
                return None
            shed = nearest_shed(pos, state.settings)
            # Extra grain can starve another worker's reserved urgent feed job.
            count = n - bag.get(item, 0)
            cost = distance(pos, shed) + 1 + distance(shed, job.location) + 1
            action = ["PICKUP", item, count] if pos == shed else move_toward(pos, shed)
            return action, cost
    cost = distance(pos, job.location) + 1
    action = list(job.action) if pos == job.location else move_toward(pos, job.location)
    return action, cost


_SERVICE_OPS = {"FEED", "WATER", "HARVEST", "COLLECT_FERTILIZER", "CARE"}


def _service_assets(state):
    assets = [
        (pos, 4 if "animal" in tile else 2)
        for pos, tile in state.tiles()
        if isinstance(tile, dict) and ("animal" in tile or tile.get("kind") == "PLANT")
    ]
    assets.sort(key=lambda row: (row[0][1], -row[0][0] if row[0][1] % 2 else row[0][0]))
    return assets


def _layout_from_assets(assets, worker_count):
    total = sum(weight for _, weight in assets)
    owners, order = {}, {}
    elapsed = 0
    for index, (pos, weight) in enumerate(assets):
        owners[pos] = min(worker_count - 1, (2 * elapsed + weight) * worker_count // (2 * total))
        order[pos] = index
        elapsed += weight
    return owners, order


def _service_layout(state, worker_count):
    """Stable soft bands: completion flags never change a tile's band."""
    return _layout_from_assets(_service_assets(state), worker_count)


def _free_grain(bag):
    return max(0, bag.get("WHEAT", 0) - sum(bag.get(item, 0) for item in ANIMALS))


def _service_route(job, unit, state):
    """A carried animal's first grain cannot fund a different existing animal."""
    if job.action == ("FEED",) and _free_grain(state.private["inventories"][unit]) < 1:
        if job.worker is not None and job.worker != unit:
            return None
        if state.private["shed"].get("WHEAT", 0) < 1:
            return None
        pos = (
            tuple(state.farm["farmer"])
            if unit == 0
            else tuple(state.farm["hands"][unit - 1])
            if unit > 0
            else state.positions[unit]
        )
        shed = nearest_shed(pos, state.settings)
        action = ["PICKUP", "WHEAT", 1] if pos == shed else move_toward(pos, shed)
        return action, distance(pos, shed) + distance(shed, job.location) + 2
    return route(job, unit, state)


def _service_visit_actions(job, jobs):
    return sum(
        other.location == job.location and bool(other.action) and other.action[0] in _SERVICE_OPS
        for other in jobs
    )


def _mandatory_suffix_fits(state, jobs, unit, used, reservations, layout):
    """Conservative mandatory tours after spending this worker's next action.

    Only real grain and observed workers count. Each mandatory action, pickup
    and Manhattan move is charged. Territories are preferences in dispatch;
    this certificate uses fixed nonoverlapping tours instead of promising an
    unbounded future scheduler or borrowing a later market purchase.
    """
    owners, order = layout
    positions = state.positions
    grains = [_free_grain(bag) for bag in state.private["inventories"]]
    shed_grain = state.private["shed"].get("WHEAT", 0)
    tours = [[] for _ in positions]
    for other in jobs:
        if not other.mandatory:
            continue
        if other.action not in (("FEED",), ("WATER",)):
            return False
        x, y = other.location
        tile = state.farm["tiles"][y][x]
        flag = "fed_today" if other.action == ("FEED",) else "watered_today"
        if not isinstance(tile, dict) or tile.get(flag, False):
            continue
        owner = reservations.get(other.identity, owners.get(other.location, unit))
        # Actual carried input may rescue a soft territory without returning
        # to an empty shed. A carried unplaced animal never supplies free grain.
        if other.action == ("FEED",) and not grains[owner] and not shed_grain:
            carriers = [i for i, grain in enumerate(grains) if grain > 0]
            if not carriers:
                return False
            owner = min(carriers, key=lambda i: (distance(positions[i], other.location), i))
        tours[owner].append(other)
    for worker, pending in enumerate(tours):
        pos = positions[worker]
        elapsed = int(worker <= unit)
        pending.sort(
            key=lambda j: (j.location != pos, order.get(j.location, 0), j.deadline, j.identity)
        )
        feeds = sum(j.action == ("FEED",) for j in pending)
        grain = grains[worker]
        for other in pending:
            if other.action == ("FEED",):
                if not grain:
                    if not shed_grain:
                        return False
                    shed = nearest_shed(pos, state.settings)
                    elapsed += distance(pos, shed) + 1
                    pos = shed
                    grain = min(4, feeds, shed_grain)
                    shed_grain -= grain
                grain -= 1
                feeds -= 1
            elapsed += distance(pos, other.location) + 1
            pos = other.location
            if state.step + elapsed - 1 > other.deadline:
                return False
    return True


def _survival_job(job, state):
    x, y = job.location
    tile = state.farm["tiles"][y][x]
    return (
        job.mandatory
        and isinstance(tile, dict)
        and (
            (job.action[0] == "WATER" and tile.get("consecutive_unwatered", 0) >= 1)
            or (
                job.action[0] == "FEED"
                and (tile.get("consecutive_unfed", 0) >= 1 or job.first_feed)
            )
        )
    )


def _reserve_critical_jobs(state, jobs, service_tours=False, maintenance_first=False):
    """Maximum-cardinality matching: deferral always has a named worker.

    Fewest eligible workers first; augmenting paths can move a reservation
    to another eligible worker so both critical jobs can be completed.
    """
    urgent = []
    options = {}
    worker_count = 1 + len(state.farm["hands"])
    for job in jobs:
        daily = maintenance_first and job.mandatory and job.action in (("FEED",), ("WATER",))
        if not _survival_job(job, state) and not daily:
            continue
        eligible = []
        for unit in range(worker_count):
            proposed = (_service_route if service_tours else route)(job, unit, state)
            if proposed is not None and state.step + proposed[1] - 1 <= job.deadline:
                eligible.append((proposed[1], unit))
        if eligible and any(job.deadline - state.step - cost + 1 <= 3 for cost, _ in eligible):
            options[job.identity] = [unit for _, unit in sorted(eligible)]
            urgent.append(job)
    worker_job = {}

    def reserve(identity, visited):
        for unit in options[identity]:
            if unit in visited:
                continue
            visited.add(unit)
            if unit not in worker_job or reserve(worker_job[unit], visited):
                worker_job[unit] = identity
                return True
        return False

    for job in sorted(
        urgent,
        key=lambda j: (
            not _survival_job(j, state),
            len(options[j.identity]),
            j.deadline,
            -j.value,
            j.identity,
        ),
    ):
        reserve(job.identity, set())
    return {identity: unit for unit, identity in worker_job.items()}


def _feed_pickup_count(
    state, job, unit, jobs, reservations, used, service_tours=False, service_visits=None
):
    """At most four grains for concrete, reachable, still-unfed animals today.

    Existing carried grain covers feasible visits before taking more from the
    shared shed. Grain travelling with an unplaced animal remains reserved for
    its first feed. Earlier workers have already acted in the current turn.
    """
    final_day = state.settings.last_action_step // state.settings.turns_per_day
    if state.day >= final_day:
        return 0

    def visit_actions(other):
        if not service_tours:
            return 1
        return (
            _service_visit_actions(other, jobs)
            if service_visits is None
            else service_visits.get(other.location, 0)
        )

    feeds = []
    for other in jobs:
        if other.action != ("FEED",):
            continue
        x, y = other.location
        tile = state.farm["tiles"][y][x]
        if isinstance(tile, dict) and "animal" in tile and not tile["fed_today"]:
            feeds.append(other)
    covered = set()
    for carrier, (pos, bag) in enumerate(
        zip(state.positions, state.private["inventories"], strict=True)
    ):
        committed = sum(bag.get(item, 0) for item in ANIMALS)
        grain = max(0, bag.get("WHEAT", 0) - committed)
        elapsed = int(carrier < unit)
        for _ in range(min(grain, len(feeds))):
            choices = []
            for other in feeds:
                owner = reservations.get(other.identity)
                if other.identity in covered or owner is not None and owner != carrier:
                    continue
                cost = distance(pos, other.location) + visit_actions(other)
                if state.step + elapsed + cost - 1 <= other.deadline:
                    choices.append((cost, other.identity, other))
            if not choices:
                break
            cost, _, other = min(choices)
            elapsed += cost
            pos = other.location
            covered.add(other.identity)
    if job.identity in covered:
        return 0
    available = state.private["shed"].get("WHEAT", 0)
    reserved_grain = 0
    for other in feeds:
        owner = reservations.get(other.identity)
        if owner is None or owner == unit:
            continue
        bag = state.private["inventories"][owner]
        committed = sum(bag.get(item, 0) for item in ANIMALS)
        reserved_grain += int(bag.get("WHEAT", 0) <= committed)
    limit = min(4, max(0, available - reserved_grain), len(feeds) - len(covered))
    if limit <= 0:
        return 0
    # This function runs only after choosing an actual shed-side PICKUP. Its
    # first target and every extra FEED include travel and one legal action.
    pos = state.positions[unit]
    elapsed = 1 + distance(pos, job.location) + visit_actions(job)
    if state.step + elapsed - 1 > job.deadline:
        if service_tours and state.step + 1 + distance(pos, job.location) <= job.deadline:
            # Optional local work may be skipped. It must never make an
            # otherwise reachable mandatory first FEED lose its real grain.
            return 1
        return 0
    count = 1
    pos = job.location
    visited = {job.identity}
    while count < limit:
        choices = []
        for other in feeds:
            owner = reservations.get(other.identity)
            if (
                other.identity in covered
                or other.identity in visited
                or other.identity in used
                or owner is not None
                and owner != unit
            ):
                continue
            cost = distance(pos, other.location) + visit_actions(other)
            if state.step + elapsed + cost - 1 <= other.deadline:
                choices.append((cost, other.identity, other))
        if not choices:
            break
        cost, _, other = min(choices)
        elapsed += cost
        pos = other.location
        visited.add(other.identity)
        count += 1
    return count


def assign(state, jobs, maintenance_first=False, service_tours=False):
    """Reserve urgent survival, then greedily assign exact legal actions.

    Complete one action per real worker. Hired-after-market workers are excluded.
    Immediate actions are applied to a private shadow to prevent double use and
    check effect; empty orders cannot make later resources magically available.
    """
    shadow = state.clone()
    actions = [["PASS"] for _ in range(1 + len(state.farm["hands"]))]
    result = Assignment(actions, shadow=shadow)
    used = set()
    reservations = _reserve_critical_jobs(
        state, jobs, service_tours=service_tours, maintenance_first=maintenance_first
    )
    layout = _service_layout(state, len(actions)) if service_tours else None
    service_visits = None
    if service_tours:
        service_visits = {}
        for job in jobs:
            if job.action and job.action[0] in _SERVICE_OPS:
                service_visits[job.location] = service_visits.get(job.location, 0) + 1
    for unit in range(len(actions)):
        choices = []
        suffix_fits = None
        pos = tuple(shadow.farm["farmer"]) if unit == 0 else tuple(shadow.farm["hands"][unit - 1])
        for job in jobs:
            if job.identity in used:
                continue
            owner = reservations.get(job.identity)
            if owner is not None and owner != unit:
                continue
            proposal = (_service_route if service_tours else route)(job, unit, shadow)
            if proposal is None:
                continue
            action, cost = proposal
            finish = state.step + cost - 1
            if finish > job.deadline:
                continue
            slack = max(0, job.deadline - finish)
            critical = _survival_job(job, shadow) and slack <= 3
            mandatory = 3.0 if job.mandatory else 1.0
            score = job.value * mandatory / cost * (1.0 + 1.0 / (slack + 1))
            if maintenance_first:
                priority = 3 if owner == unit else 2 if critical else int(job.mandatory)
            else:
                priority = 2 if owner == unit else int(critical)
            continuation = territory = sequence = 0
            if service_tours:
                final_day = state.settings.last_action_step // state.settings.turns_per_day
                if job.identity.startswith("deliver:") and state.day == final_day and slack <= 1:
                    priority = max(priority, 4)
                if job.action and job.action[0] in _SERVICE_OPS:
                    local = job.location == pos and action[0] == job.action[0]
                    if local and not job.mandatory and priority < 2:
                        if suffix_fits is None:
                            suffix_fits = _mandatory_suffix_fits(
                                shadow, jobs, unit, used, reservations, layout
                            )
                        x, y = job.location
                        tile = shadow.farm["tiles"][y][x]
                        needs_basics = isinstance(tile, dict) and (
                            "animal" in tile
                            and not tile.get("fed_today", False)
                            or tile.get("kind") == "PLANT"
                            and not tile.get("watered_today", False)
                            and any(j.location == pos and j.action == ("WATER",) for j in jobs)
                        )
                        if suffix_fits and not needs_basics:
                            priority = max(priority, 1)
                    if priority >= 1:
                        continuation = int(local)
                        territory = int(layout[0].get(job.location) == unit)
                        if job.action == ("FEED",) and _free_grain(
                            shadow.private["inventories"][unit]
                        ):
                            assigned = layout[0].get(job.location, unit)
                            if not _free_grain(
                                shadow.private["inventories"][assigned]
                            ) and not shadow.private["shed"].get("WHEAT", 0):
                                territory = 2
                        sequence = -layout[1].get(job.location, 0)
            choices.append(
                (
                    priority,
                    continuation,
                    territory,
                    sequence,
                    score,
                    job.identity,
                    job,
                    action,
                    cost,
                )
            )
        choices.sort(key=lambda row: (-row[0], -row[1], -row[2], -row[3], -row[4], row[5]))
        for _, _, _, _, _, _, job, action, cost in choices:
            if job.batch_feed and action[:2] == ["PICKUP", "WHEAT"]:
                count = _feed_pickup_count(
                    shadow, job, unit, jobs, reservations, used, service_tours, service_visits
                )
                if not count:
                    continue  # Carried grain or another urgent worker owns this feed input.
                action = ["PICKUP", "WHEAT", count]
            # Pinned apply_unit is atomic on failure except DROP overflow and
            # missing-inventory supplementation. Those retain full rollback.
            before = (
                shadow.clone()
                if (action and action[0] == "DROP") or len(shadow.private["inventories"]) <= unit
                else None
            )
            effect, discarded = apply_unit(shadow, unit, action)
            if effect is None or discarded:
                if before is not None:
                    shadow = before
                    result.shadow = shadow
                result.rejected.append(
                    dict(
                        unit=unit,
                        job=job.identity,
                        action=action,
                        reason="no effect or overflow in exact reservation",
                    )
                )
                reservations.pop(job.identity, None)
                continue
            used.add(job.identity)
            actions[unit] = action
            result.selected.append(
                dict(
                    unit=unit,
                    job=job.identity,
                    action=action,
                    estimated_remaining_actions=cost,
                    deadline=job.deadline,
                    value=job.value,
                    prerequisites=list(job.requires),
                    reason=job.reason,
                    **(
                        dict(
                            service_territory=layout[0].get(job.location),
                            service_continuation=job.location == pos,
                            mandatory_suffix_fits=suffix_fits,
                        )
                        if service_tours
                        else {}
                    ),
                )
            )
            break
    return result


def schedule_value(state, jobs, extra_workers=0, service_tours=False, prepared=None):
    """Approximate achievable daily value; used only for marginal hiring.

    Travel/prerequisite duration is included. Sequential dependencies between
    different jobs are conservatively left independent, so this is not exact.
    """
    if service_tours:
        return _service_schedule_value(state, jobs, extra_workers, prepared)
    positions = state.positions
    worker_count = len(positions)
    shed_tiles = state.settings.shed_tiles()
    positions += [shed_tiles[i % 4] for i in range(extra_workers)]
    clocks = [state.step + (1 if i >= worker_count else 0) for i in range(len(positions))]
    value = 0.0
    inventories = state.private["inventories"]
    for job in sorted(jobs, key=lambda j: (not j.mandatory, j.deadline, -j.value)):
        carry = tuple((item, n) for item, n, source in job.requires if source == "carry")
        best = None
        for unit, pos in enumerate(positions):
            if job.worker is not None and job.worker != unit:
                continue
            cost = distance(pos, job.location) + 1
            if carry:
                bag = inventories[unit] if unit < worker_count else {}
                if any(bag.get(item, 0) < n for item, n in carry):
                    access = nearest_shed(pos, state.settings)
                    cost = distance(pos, access) + 1 + distance(access, job.location) + 1
            end = clocks[unit] + cost - 1
            if end <= job.deadline:
                choice = (end, unit)
                if best is None or choice < best:
                    best = choice
        if best is not None:
            end, unit = best
            clocks[unit] = end + 1
            positions[unit] = job.location
            value += job.value
    return value


def _prepare_service_schedule(state, jobs):
    """Crew-independent geometry/order only; no clocks, prices or resources."""
    assets = _service_assets(state)
    order = {pos: index for index, (pos, _) in enumerate(assets)}
    grouped = {}
    for job in jobs:
        key = (
            ("service", job.location)
            if job.action and job.action[0] in _SERVICE_OPS
            else ("single", job.identity)
        )
        grouped.setdefault(key, []).append(job)
    groups = sorted(
        grouped.values(),
        key=lambda group: (
            not any(j.mandatory for j in group),
            min(j.deadline for j in group),
            order.get(group[0].location, len(order)),
            group[0].identity,
        ),
    )
    bundles = tuple(
        tuple(sorted(group, key=lambda j: (not j.mandatory, j.deadline, -j.value, j.identity)))
        for group in groups
    )
    return assets, bundles, sum(j.action == ("FEED",) for j in jobs)


def _service_schedule_value(state, jobs, extra_workers, prepared=None):
    """Bounded hiring estimate: one visit per bundle, real shared inputs.

    This is a capacity estimate, not a future state. Concrete next-turn work
    is still proved by the same allocator used by live play and rollouts.
    """
    positions = state.positions
    worker_count = len(positions)
    for _ in range(max(0, extra_workers)):
        tiles = state.settings.shed_tiles()
        positions.append(min(tiles, key=lambda p: (positions.count(p), tiles.index(p))))
    clocks = [state.step + int(i >= worker_count) for i in range(len(positions))]
    bags = [bag.copy() for bag in state.private["inventories"]]
    bags.extend({} for _ in range(max(0, extra_workers)))
    shed = state.private["shed"].copy()
    seeds = state.private["seeds"].copy()
    assets, bundles, remaining_feeds = (
        _prepare_service_schedule(state, jobs) if prepared is None else prepared
    )
    owners, _ = _layout_from_assets(assets, len(positions))
    value = 0.0
    for group in bundles:
        if len(group) == 1 and not group[0].requires:
            job = group[0]
            best = None
            for unit, start in enumerate(positions):
                if job.worker is not None and job.worker != unit:
                    continue
                end = clocks[unit] + distance(start, job.location)
                if end <= job.deadline:
                    earned = 0.0 + job.value
                    if earned:
                        choice = (earned, owners.get(job.location) == unit, -(end + 1), -unit)
                        if best is None or choice > best[0]:
                            best = choice, unit, end + 1
            if best is not None:
                choice, unit, clock = best
                positions[unit], clocks[unit] = job.location, clock
                remaining_feeds -= int(job.action == ("FEED",))
                value += choice[0]
            # No resource dictionary changes anywhere in a single no-input job.
            continue
        choices = []
        for unit, start in enumerate(positions):
            bag, stock, available_seeds = bags[unit].copy(), shed.copy(), seeds.copy()
            pos, clock, earned, feeds = start, clocks[unit], 0.0, 0
            for job in group:
                if job.worker is not None and job.worker != unit:
                    continue
                if not job.requires:
                    end = clock + distance(pos, job.location)
                    if end <= job.deadline:
                        pos, clock = job.location, end + 1
                        earned += job.value
                        feeds += int(job.action == ("FEED",))
                    # No resources change on this estimate's no-input jobs.
                    # Later input jobs still transact on their own copies.
                    continue
                candidate_pos, candidate_clock = pos, clock
                candidate_bag, candidate_stock = bag.copy(), stock.copy()
                legal = True
                for item, count, source in job.requires:
                    if source == "seed":
                        if available_seeds.get(item, 0) < count:
                            legal = False
                            break
                    else:
                        held = (
                            _free_grain(candidate_bag)
                            if job.action == ("FEED",) and item == "WHEAT"
                            else candidate_bag.get(item, 0)
                        )
                        missing = max(0, count - held)
                        if not missing:
                            continue
                        if candidate_stock.get(item, 0) < missing:
                            legal = False
                            break
                        access = nearest_shed(candidate_pos, state.settings)
                        candidate_clock += distance(candidate_pos, access) + 1
                        candidate_pos = access
                        take = missing
                        if job.action == ("FEED",) and item == "WHEAT":
                            take = min(4, remaining_feeds, candidate_stock[item])
                        candidate_stock[item] -= take
                        candidate_bag[item] = candidate_bag.get(item, 0) + take
                end = candidate_clock + distance(candidate_pos, job.location)
                if not legal or end > job.deadline:
                    continue
                candidate_clock = end + 1
                for item, count, source in job.requires:
                    if source == "seed":
                        available_seeds[item] -= count
                    elif not (job.action and job.action[0] == "PLACE" and item == "WHEAT"):
                        candidate_bag[item] -= count
                pos, clock, bag, stock = (
                    job.location,
                    candidate_clock,
                    candidate_bag,
                    candidate_stock,
                )
                earned += job.value
                feeds += int(job.action == ("FEED",))
            if earned:
                choices.append((earned, clock, unit, pos, bag, stock, available_seeds, feeds))
        if choices:
            # A soft band cannot suppress feasible capacity elsewhere.
            best = max(
                choices,
                key=lambda row: (row[0], owners.get(group[0].location) == row[2], -row[1], -row[2]),
            )
            earned, clock, unit, pos, bag, shed, seeds, feeds = best
            positions[unit], clocks[unit], bags[unit] = pos, clock, bag
            remaining_feeds -= feeds
            value += earned
    return value
