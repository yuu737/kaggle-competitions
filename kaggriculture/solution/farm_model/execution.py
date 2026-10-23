"""One executor for live play and planner branches. No simulated state is global."""

from dataclasses import dataclass

from .forecast import known_demand, running_reserve
from .jobs import _prepare_service_schedule, assign, generate_jobs, schedule_value
from .market import execute_market, quote_purchase, quote_sale
from .physics import advance
from .projects import input_counts, installed, unique_projects
from .rules import ANIMALS, CROPS, LAND_COSTS, PRODUCTS, hire_cost


@dataclass(frozen=True)
class Policy:
    max_workers: int = 10
    max_assets: int = 20
    search_seconds: float = 0.20
    rollout_turns: int = 24
    candidate_limit: int = 8
    physical_candidates: int = 2
    minimum_gain: float = 25.0
    investments: bool = True
    adaptive_sales: bool = False
    hiring: bool = True
    allowed_items: tuple = ()
    investment_strategy: str = "daily"
    farm_planning: bool = False
    staged_projects: int = 2
    service_tours: bool = False
    large_batches: bool = False
    cash_bridge: bool = False
    rival_cycles: bool = False
    rotation_only: bool = False
    preserve_rotations: bool = False
    additive_growth: bool = False


def sales_to_make(state, policy, added_animals=0):
    animals = len(state.animals()) + added_animals
    final_day = state.settings.last_action_step // state.settings.turns_per_day
    keep_feed = animals * 2 if state.day < final_day else 0
    carried_feed = sum(bag.get("WHEAT", 0) for bag in state.private["inventories"])
    keep_feed = max(0, keep_feed - carried_feed)
    room = state.settings.shed_capacity - state.shed_used()
    hold_budget = max(0, min(state.settings.shed_capacity // 4, room))
    planned = []
    decisions = []
    for item in PRODUCTS:
        count = state.private["shed"].get(item, 0)
        if not count:
            continue
        keep = min(count, keep_feed) if item == "WHEAT" else 0
        reason = "feed reserve" if keep else "immediate cash"
        free = count - keep
        if (
            free
            and policy.adaptive_sales
            and state.day < final_day - 1
            and state.farm["money"] > running_reserve(state)
            and hold_budget > 0
            and item != "FERTILIZER"
        ):
            demand = known_demand(state, state.day + 1)[item]
            public_supply = 0.0
            for _, tile in state.tiles(opponent=True):
                if isinstance(tile, dict) and tile.get("animal") in ANIMALS:
                    animal = ANIMALS[tile["animal"]]
                    if animal["product"] == item:
                        public_supply += 1.5 * (1.0 + animal["interval"]) / animal["interval"]
            now, _ = quote_sale(item, free, state.market[item], state.params)
            later, _ = quote_sale(
                item, free, state.market[item] - demand + public_supply, state.params
            )
            if later > now * 1.10 + 2 * free:
                hold = min(free, hold_budget, 10)
                keep += hold
                hold_budget -= hold
                reason = "bounded one-day recovery hypothesis"
        sell = count - keep
        if sell:
            planned.append(["SELL", item, sell])
        decisions.append(dict(item=item, held=keep, sold=sell, reason=reason))
    # Large deposits first relieve capacity; order is preserved in execution.
    planned.sort(key=lambda order: (-order[2], order[1]))
    return planned, decisions


class OrderBuilder:
    def __init__(self, state):
        self.state = state.clone()
        self.orders = []
        self.trace = []
        # Only the $1 floor is guaranteed for sale-funded purchases when the
        # opposing simultaneous order is unknown. Do not borrow forecast cash.
        self.guaranteed_cash = float(state.farm["money"])

    def append(self, order, cash_bound=0.0, reserve=0.0):
        if len(self.orders) >= self.state.settings.max_orders:
            self.trace.append(dict(order=order, accepted=False, reason="order limit"))
            return False
        if order[0] != "SELL" and self.guaranteed_cash - cash_bound < reserve:
            self.trace.append(dict(order=order, accepted=False, reason="cash runway"))
            return False
        branch = self.state.clone()
        own = dict(farm=branch.farm, private=branch.private)
        dummy = dict(farm=branch.opponent, private=dict(shed={}, seeds={}, inventories=[]))
        accounts = [own, dummy] if branch.player == 0 else [dummy, own]
        queues = [[order], []] if branch.player == 0 else [[], [order]]
        trace = execute_market(accounts, branch.market, queues, branch.settings, branch.params)
        expected = order[2] if len(order) >= 3 else 1
        executed = sum(r["executed"] for r in trace if r["seat"] == branch.player)
        if executed != expected:
            self.trace.append(
                dict(order=order, accepted=False, reason="partial/failed exact execution")
            )
            return False
        self.state = branch
        self.orders.append(order)
        self.guaranteed_cash += expected if order[0] == "SELL" else -cash_bound
        self.trace.append(dict(order=order, accepted=True, guaranteed_cash=self.guaranteed_cash))
        return True


def new_hand_has_job(
    state,
    projects,
    maintenance_first=False,
    prepared_livestock=False,
    job_cache=None,
    service_tours=False,
):
    """Require a concrete next-turn assignment before paying a hire's wage.

    Use the official spawn/market/decay order and the real allocator. A coarse
    capacity estimate alone can promise a job that earlier workers always take.
    state is already after current worker actions and prior market orders.
    """
    trial = state.clone()
    old_workers = len(trial.positions)
    advance(
        trial,
        dict(farmer=["PASS"], hands=[["PASS"]] * (old_workers - 1), market=[["HIRE"]]),
    )
    if len(trial.positions) != old_workers + 1:
        return False
    final_day = trial.settings.last_action_step // trial.settings.turns_per_day
    next_jobs = job_cache.get("next") if job_cache is not None else None
    if next_jobs is None:
        next_jobs = generate_jobs(trial, projects, prepared_livestock=prepared_livestock)
        if job_cache is not None:
            job_cache["next"] = next_jobs
    allocation = assign(
        trial,
        next_jobs,
        maintenance_first=maintenance_first and trial.day < final_day,
        service_tours=service_tours,
    )
    return allocation.actions[-1] != ["PASS"]


def farm_cash_reserve(state, added_animals=0, max_workers=10):
    """Cash still needed for maintenance, crediting observed grain only once.

    Start with the existing wage model, replace its whole-feed cash estimate
    with the missing grain quantity's conservative market bound. Grain in the
    shed and carry is a single shared balance, not a separate credit for every
    future animal. Recompute after a successful feed order from builder.state.
    """
    animals = len(state.animals()) + added_animals
    modeled_feed, _ = quote_purchase("WHEAT", 2 * animals, state.market["WHEAT"], state.params)
    wages = max(0.0, running_reserve(state, added_animals, max_workers) - modeled_feed)
    final_day = state.settings.last_action_step // state.settings.turns_per_day
    needed = 2 * animals if state.day < final_day else 0
    grain = state.private["shed"].get("WHEAT", 0) + sum(
        bag.get("WHEAT", 0) for bag in state.private["inventories"]
    )
    missing = max(0, needed - grain)
    rival_bound = state.settings.max_orders * state.settings.shed_capacity
    feed, _ = quote_purchase("WHEAT", missing, state.market["WHEAT"] - rival_bound, state.params)
    return wages + feed


def current_feed_reserve(state):
    """Guaranteed cash for today's remaining feeds after real worker actions.

    Existing animals fed by the current action are excluded. Shed and carry
    share one grain balance; neither future sales nor unplaced intentions can
    fund a maintenance hire. Tomorrow's runway still guards optional purchases.
    """
    final_day = state.settings.last_action_step // state.settings.turns_per_day
    if state.day >= final_day:
        return 0.0
    unfed = sum(not tile["fed_today"] for _, tile in state.animals())
    grain = state.private["shed"].get("WHEAT", 0) + sum(
        bag.get("WHEAT", 0) for bag in state.private["inventories"]
    )
    missing = max(0, unfed - grain)
    rival_bound = state.settings.max_orders * state.settings.shed_capacity
    feed, _ = quote_purchase("WHEAT", missing, state.market["WHEAT"] - rival_bound, state.params)
    return feed


def stage_projects(state, projects, limit, max_workers=10):
    """Activate real-input contracts, then a bounded number needing purchases.

    Empty livestock structures get priority when assigning their observed
    animals or selecting the next purchase. A barn does not itself count as an
    owned animal and cannot bypass the unfunded-contract limit.
    """
    projects = unique_projects(state, projects)
    available = input_counts(state)
    built, other = [], []
    for project in projects:
        x, y = project.location
        tile = state.farm["tiles"][y][x]
        ready_barn = (
            project.animal
            and isinstance(tile, dict)
            and tile.get("kind") == ANIMALS[project.item]["structure"]
        )
        (built if ready_barn else other).append(project)
    funded, unfunded = [], []
    for project in [*built, *other]:
        if available[project.item] > 0:
            available[project.item] -= 1
            funded.append(project)
        else:
            unfunded.append(project)
    accepted = []
    land_reserved = any(
        state.farm["tiles"][p.location[1]][p.location[0]] == "LOCKED" for p in funded
    )
    index = len(state.farm["unlocked_quadrants"]) - 1
    capital = float(LAND_COSTS[index]) if land_reserved else 0.0
    added_animals = sum(project.animal for project in funded)
    for project in unfunded:
        if len(accepted) >= max(0, int(limit)):
            break
        unit_cost = ANIMALS[project.item]["cost"] if project.animal else CROPS[project.item]["seed"]
        x, y = project.location
        needs_land = state.farm["tiles"][y][x] == "LOCKED"
        index = len(state.farm["unlocked_quadrants"]) - 1
        land_cost = LAND_COSTS[index] if needs_land and not land_reserved else 0
        new_animals = added_animals + int(project.animal)
        reserve = farm_cash_reserve(state, new_animals, max_workers)
        if capital + unit_cost + land_cost + reserve > state.farm["money"]:
            continue
        accepted.append(project)
        capital += unit_cost + land_cost
        added_animals = new_animals
        land_reserved = land_reserved or needs_land
    return tuple([*funded, *accepted])


def _capacity_value(crew, jobs, extra_workers, policy, job_cache):
    prepared = job_cache.get("prepared") if job_cache is not None else None
    if policy.service_tours and job_cache is not None and prepared is None:
        # This cache belongs to one HIRE-only batch. Time, assets and jobs stay
        # fixed; positions, input balances and clocks are recalculated below.
        prepared = _prepare_service_schedule(crew, jobs)
        job_cache["prepared"] = prepared
    return schedule_value(
        crew, jobs, extra_workers, service_tours=policy.service_tours, prepared=prepared
    )


def hire_for_work(
    builder, after_workers, projects, policy, maintenance_first, reserve, job_cache=None
):
    """Fund one useful hire from guaranteed cash after today's feed obligation."""
    if (
        not policy.hiring
        or len(builder.state.positions) >= policy.max_workers
        or after_workers.end_of_day_step - after_workers.step < 4
    ):
        return False
    cost = hire_cost(builder.state.farm["hires_today"], after_workers.settings.hire_mult)
    hire_reserve = current_feed_reserve(builder.state) if policy.farm_planning else reserve
    if policy.farm_planning and builder.guaranteed_cash < cost + hire_reserve:
        return False
    # Earlier hires are real members of the next-turn crew. Re-evaluate the
    # marginal job capacity with that crew instead of pricing every new hand
    # against the original single worker.
    crew = builder.state if policy.farm_planning else after_workers
    remaining_jobs = job_cache.get("current") if job_cache is not None else None
    if remaining_jobs is None:
        remaining_jobs = generate_jobs(crew, projects, prepared_livestock=policy.farm_planning)
        if job_cache is not None:
            job_cache["current"] = remaining_jobs
    improvement = _capacity_value(crew, remaining_jobs, 1, policy, job_cache) - _capacity_value(
        crew, remaining_jobs, 0, policy, job_cache
    )
    if improvement > max(1.0, 1.2 * cost) and new_hand_has_job(
        builder.state,
        projects,
        maintenance_first=maintenance_first,
        prepared_livestock=policy.farm_planning,
        job_cache=job_cache,
        service_tours=policy.service_tours,
    ):
        return builder.append(["HIRE"], cost, hire_reserve)
    return False


def hire_batch_for_work(
    builder, after_workers, projects, policy, maintenance_first, reserve, slots, capacity_cache=None
):
    """Buy only marginal, concretely assigned hands while preserving order slots."""
    hired = 0
    # This scope contains HIRE orders only. Assets, market prices, carried
    # goods, time and project inputs are fixed; each spawn adds an empty bag.
    # Re-entering after a feed purchase creates a fresh cache.
    job_cache = {"capacity": capacity_cache} if capacity_cache is not None else {}
    if capacity_cache is not None:
        capacity_cache["batch_failed_without_effect"] = False
    while len(builder.orders) < builder.state.settings.max_orders - slots:
        trace_size = len(builder.trace)
        if not hire_for_work(
            builder, after_workers, projects, policy, maintenance_first, reserve, job_cache
        ):
            if capacity_cache is not None:
                capacity_cache["batch_failed_without_effect"] = len(builder.trace) == trace_size
            break
        hired += 1
    return hired


def execute(state, projects=(), policy=None):
    policy = policy or Policy()
    planned_projects = unique_projects(state, projects)
    projects = (
        stage_projects(state, planned_projects, policy.staged_projects, policy.max_workers)
        if policy.farm_planning
        else planned_projects
    )
    cfg = state.settings
    final_day = cfg.last_action_step // cfg.turns_per_day
    maintenance_first = policy.farm_planning and state.day < final_day
    jobs = generate_jobs(state, projects, prepared_livestock=policy.farm_planning)
    allocation = assign(
        state, jobs, maintenance_first=maintenance_first, service_tours=policy.service_tours
    )
    after_workers = allocation.shadow
    builder = OrderBuilder(after_workers)
    animals = len(after_workers.animals())
    pending_animals = sum(
        project.animal and not installed(after_workers, project) for project in projects
    )
    total_feed = after_workers.private["shed"].get("WHEAT", 0) + sum(
        bag.get("WHEAT", 0) for bag in after_workers.private["inventories"]
    )
    target_feed = 2 * (animals + pending_animals) if state.day < final_day else 0
    buy_feed = max(0, target_feed - total_feed)
    reserve = (
        farm_cash_reserve(after_workers, pending_animals, policy.max_workers)
        if policy.farm_planning
        else running_reserve(after_workers, pending_animals)
    )
    sales, decisions = sales_to_make(
        after_workers, policy, pending_animals if policy.farm_planning else 0
    )
    # Leave market capacity for today's crew as well as maintenance inputs and
    # committed projects. One early sale still provides a guaranteed cash floor.
    if policy.farm_planning:
        project_slots = len({p.item for p in projects}) + int(any(p.needs_land for p in projects))
    else:
        project_slots = int(bool(projects))
    procurement_slots = int(buy_feed > 0) + project_slots
    hire_slots = int(policy.hiring)
    if policy.farm_planning and policy.hiring:
        hire_slots = min(
            max(0, policy.max_workers - len(after_workers.positions)),
            max(0, cfg.max_orders - procurement_slots - int(bool(sales))),
        )
    critical_slots = procurement_slots + hire_slots
    first_sales = max(0, cfg.max_orders - critical_slots)
    for order in sales[:first_sales]:
        builder.append(order)
    # Actual maintenance staffing precedes tomorrow's speculative feed buffer.
    # Its present-day missing grain cost is protected by the same cash ledger.
    hired = 0
    capacity_cache = {} if policy.service_tours else None
    if policy.farm_planning:
        hired = hire_batch_for_work(
            builder,
            after_workers,
            projects,
            policy,
            maintenance_first,
            reserve,
            procurement_slots,
            capacity_cache=capacity_cache,
        )
    if buy_feed:
        room = cfg.shed_capacity - builder.state.shed_used()
        count = min(buy_feed, room)
        # Each opposing non-seed purchase deposits into a bounded shed. Even
        # sell/buy cycling cannot remove more than orders * capacity this turn.
        # Ignore our wheat-sale deposits when bounding cost; they only help.
        rival_purchase_bound = cfg.max_orders * cfg.shed_capacity
        while count > 0:
            bound, _ = quote_purchase(
                "WHEAT", count, after_workers.market["WHEAT"] - rival_purchase_bound, state.params
            )
            if builder.guaranteed_cash >= bound:
                break
            count -= 1
        if count:
            builder.append(["BUY_PRODUCT", "WHEAT", count], bound)
    if policy.farm_planning:
        # A filled feed order supplies real grain for the committed heads.
        # Charging its cash cost again would strand an otherwise funded animal.
        reserve = farm_cash_reserve(builder.state, pending_animals, policy.max_workers)
    repeat_hire_failure = (
        capacity_cache is not None
        and not buy_feed
        and capacity_cache.get("batch_failed_without_effect", False)
    )
    if policy.farm_planning and not repeat_hire_failure:
        # Feed bought after staffing may unlock real next-turn assignments.
        # Existing hires count toward the capacity calculation on this pass.
        hire_batch_for_work(
            builder,
            after_workers,
            projects,
            policy,
            maintenance_first,
            reserve,
            project_slots,
            capacity_cache=capacity_cache,
        )
    elif not policy.farm_planning and not hired:
        hire_for_work(builder, after_workers, projects, policy, maintenance_first, reserve)
    remaining = [p for p in projects if not installed(after_workers, p)]
    # All feasible locked sites belong to the next quadrant in the authoritative
    # observation. A batch has at most one land purchase, never a chain of buys.
    locked = [
        p
        for p in remaining
        if p.needs_land and after_workers.farm["tiles"][p.location[1]][p.location[0]] == "LOCKED"
    ]
    index = len(after_workers.farm["unlocked_quadrants"]) - 1
    if locked and index < len(LAND_COSTS):
        builder.append(["BUY_LAND"], LAND_COSTS[index], reserve)
    requirements = {}
    for project in remaining:
        x, y = project.location
        tile = builder.state.farm["tiles"][y][x]
        if (
            policy.farm_planning
            and project.animal
            and not (
                isinstance(tile, dict)
                and tile.get("kind") == ANIMALS[project.item]["structure"]
                and "animal" not in tile
            )
        ):
            # Animal capital follows completed physical preparation. A BUILD
            # by this turn's old crew is visible here, but new purchases cannot
            # participate in the already completed worker actions.
            continue
        if tile != "LOCKED":
            requirements[project.item] = requirements.get(project.item, 0) + 1
    available = input_counts(after_workers)
    for item, needed in requirements.items():
        missing = max(0, needed - available[item])
        if not missing:
            continue
        unit_cost = ANIMALS[item]["cost"] if item in ANIMALS else CROPS[item]["seed"]
        affordable = max(0, int((builder.guaranteed_cash - reserve) // unit_cost))
        count = min(missing, affordable)
        if item in ANIMALS:
            count = min(count, max(0, cfg.shed_capacity - builder.state.shed_used()))
        if count:
            op = "BUY_ANIMAL" if item in ANIMALS else "BUY_SEED"
            builder.append([op, item, count], count * unit_cost, reserve)
    for order in sales[first_sales:]:
        builder.append(order)
    action = dict(farmer=allocation.actions[0], hands=allocation.actions[1:], market=builder.orders)
    return action, dict(
        assignments=allocation.selected,
        rejected_jobs=allocation.rejected,
        market=builder.trace,
        sales=decisions,
        feed_target=target_feed,
        cash_reserve=reserve,
        projects=[
            dict(item=p.item, location=p.location, needs_land=p.needs_land) for p in projects
        ],
        deferred_projects=len(planned_projects) - len(projects),
    )
