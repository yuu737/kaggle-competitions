"""Approximate season cash ledger, separated from exact game transitions.

It is intentionally an inspectable hypothesis: service, opponent deliveries,
unknown future shops and daily trade timing are approximations. Physical first-
day rollout and official matches, not this forecast, decide correctness/strength.
"""

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from math import ceil
from time import perf_counter

from .market import quote_purchase, quote_sale
from .physics import refresh_tile
from .projects import Project, available_sites, can_install
from .rules import (
    ANIMALS,
    CROPS,
    LAND_COSTS,
    PRODUCTS,
    SHOPS,
    hire_cost,
    new_animal,
    new_plant,
)
from .state import distance, nearest_shed


@dataclass(frozen=True)
class Scenario:
    name: str
    opponent_supply: float
    unknown_demand: float
    service: float


SCENARIOS = (
    Scenario("adverse", 1.5, 0.0, 0.85),
    Scenario("central", 1.0, 1.0, 0.95),
    Scenario("favorable", 0.5, 1.25, 1.0),
)


@dataclass
class Forecast:
    terminal_cash: float
    minimum_cash: float
    sales: float
    feed_cost: float
    wages: float
    capital_cost: float
    worker_days: int
    rows: list
    feasible: bool
    assumptions: dict


def known_demand(state, day, shops=None):
    cfg = state.settings
    start = max(state.step, day * cfg.turns_per_day)
    end = min(cfg.last_action_step, (day + 1) * cfg.turns_per_day - 1)
    result = {item: 0.0 for item in PRODUCTS}
    for turn in range(start, end + 1):
        if turn % cfg.shop_interval == 0:
            for shop in state.shops if shops is None else shops:
                products = SHOPS[shop]
                for item in products:
                    result[item] += 2 if len(products) == 1 else 1
        if turn % cfg.center_interval == 0:
            for item in PRODUCTS:
                if item != "FERTILIZER":
                    result[item] += 1
    return result


def demand_forecast(state, day, scenario):
    result = known_demand(state, day)
    # Uniform expectation over the official eight shop types. The random seed,
    # actual future shop draws, and replay futures are never inputs.
    future_unlocks = max(
        0, day // state.settings.unlock_interval - state.day // state.settings.unlock_interval
    )
    future_unlocks = min(future_unlocks, max(0, 8 - len(state.shops)))
    ticks = state.settings.turns_per_day / state.settings.shop_interval
    for products in SHOPS.values():
        for item in products:
            result[item] += (
                future_unlocks
                * ticks
                / len(SHOPS)
                * (2 if len(products) == 1 else 1)
                * scenario.unknown_demand
            )
    return result


def _daily_outputs(assets, day, settings, service=1.0, care_fraction=1.0):
    """Ideal scheduled maintenance, explicit stock caps, harvest and decay.

    service reduces *monetizable* output; it does not promise perfect maintenance.
    care_fraction changes future care banking only. Care already observed today
    still contributes its full unit after the official production/reset order.
    The physical rollout independently checks the actual maintenance schedule.
    """
    output = {item: 0.0 for item in PRODUCTS}
    feed = 0
    active = []
    for pos, tile in assets:
        if "animal" in tile:
            spec = ANIMALS[tile["animal"]]
            already_cared = tile["cared_today"]
            output[spec["product"]] += tile["yield_units"] * service
            tile["yield_units"] = 0
            if tile["fertilizer_available"]:
                output["FERTILIZER"] += service
                tile["fertilizer_available"] = False
            feed += 1
            tile["fed_today"] = tile["cared_today"] = True
            tile = refresh_tile(tile, day, settings)
            if not already_cared and care_fraction != 1.0:
                # refresh_tile produces using the prior bank, resets it on a
                # production event, then adds today's unit. Replace only that
                # new unit; previously earned care and yielded goods are real.
                tile["pending_care_bonus"] -= 1.0 - care_fraction
            active.append((pos, tile))
        elif tile.get("kind") == "PLANT":
            spec = CROPS[tile["crop"]]
            age = day - tile["planted_day"]
            if not tile["watered_today"]:
                tile["watered_today"] = True
                if not spec["interval"] and (spec["peak"] + 1) // 2 <= age <= spec["peak"]:
                    tile["yield_units"] = min(spec["cap"], tile["yield_units"] + 1)
            peak = min(spec["peak"], 10) if tile["crop"] == "MELON" else spec["peak"]
            final_day = settings.last_action_step // settings.turns_per_day
            harvest = (
                age >= spec["first"]
                and tile["yield_units"] > 0
                and (spec["interval"] or age >= peak or day == final_day)
            )
            if harvest:
                output[tile["crop"]] += tile["yield_units"] * service
                tile["yield_units"] = 0
                if not spec["interval"]:
                    continue
            life = tile.get("max_lifespan_step", -1)
            if life >= 0 and day * settings.turns_per_day >= life:
                continue
            tile = refresh_tile(tile, day, settings)
            if tile.get("kind") == "PLANT":
                active.append((pos, tile))
    return active, output, feed


def workload(assets, settings):
    """Daily action count plus a nearest-neighbor travel tour (approximate)."""
    if not assets:
        return 0.0
    remaining = [pos for pos, _ in assets]
    pos = settings.shed_tiles()[0]
    travel = 0
    while remaining:
        target = min(remaining, key=lambda p: (distance(pos, p), p))
        travel += distance(pos, target)
        remaining.remove(target)
        pos = target
    travel += distance(pos, nearest_shed(pos, settings))
    actions = 2.0  # pickup/delivery
    for _, tile in assets:
        if "animal" in tile:
            actions += 3.0 + 1.0 / ANIMALS[tile["animal"]]["interval"]
        elif tile.get("kind") == "PLANT":
            actions += 1.0 + (1.0 / max(1, CROPS[tile["crop"]]["interval"] or 4))
    return travel + actions


def forecast(state, scenario, project=None, max_workers=10):
    cfg = state.settings
    if state.step > cfg.last_action_step:
        # After the final market phase every remaining non-cash holding has
        # exactly zero terminal value. Do not invent another daily sale.
        cash = float(state.farm["money"])
        return Forecast(
            cash,
            cash,
            0.0,
            0.0,
            0.0,
            0.0,
            0,
            [],
            project is None,
            dict(
                scenario=scenario.name,
                horizon="finished",
                asset_terminal_value=0,
                unliquidated_inventory_value=0,
            ),
        )
    assets = [
        (pos, deepcopy(tile))
        for pos, tile in state.tiles()
        if isinstance(tile, dict) and ("animal" in tile or tile.get("kind") == "PLANT")
    ]
    opponent_assets = [
        (pos, deepcopy(tile))
        for pos, tile in state.tiles(opponent=True)
        if isinstance(tile, dict) and ("animal" in tile or tile.get("kind") == "PLANT")
    ]
    market = dict(state.market)
    cash = float(state.farm["money"])
    capital = 0.0
    pending_day = None
    if project is not None:
        pos = project.location
        access = nearest_shed(pos, cfg)
        preparation = distance(access, pos) + (4 if project.animal else 3)
        pending_day = (state.step + preparation) // cfg.turns_per_day
        capital = ANIMALS[project.item]["cost"] if project.animal else CROPS[project.item]["seed"]
        if project.needs_land:
            index = len(state.farm["unlocked_quadrants"]) - 1
            capital += LAND_COSTS[index]
        cash -= capital
    minimum = cash
    totals = dict(sales=0.0, feed_cost=0.0, wages=0.0, worker_days=0)
    rows = []
    # Existing cargo is not cash. Here it is assumed delivered; physical rollout
    # verifies this for the first day and endgame tests cover exact timing.
    holdings = {
        item: state.private["shed"].get(item, 0)
        + sum(bag.get(item, 0) for bag in state.private["inventories"])
        for item in PRODUCTS
    }
    fractional = {item: 0.0 for item in PRODUCTS}
    last_day = cfg.last_action_step // cfg.turns_per_day
    feasible = True
    for day in range(state.day, last_day + 1):
        if pending_day == day:
            tile = (
                new_animal(project.item, day)
                if project.animal
                else new_plant(project.item, day, cfg.turns_per_day)
            )
            assets.append((project.location, tile))
        today_work = workload(assets, cfg)
        # 70% usable time allows pickup, reset travel and imperfect assignment.
        workers = max(1, ceil(today_work / max(1, cfg.turns_per_day * 0.70)))
        if workers > max_workers:
            feasible = False
        workers = min(max_workers, workers)
        wages = sum(hire_cost(i, cfg.hire_mult) for i in range(workers - 1))
        if day == state.day:
            wages = sum(
                hire_cost(i, cfg.hire_mult)
                for i in range(
                    state.farm["hires_today"], max(state.farm["hires_today"], workers - 1)
                )
            )
        cash -= wages
        totals["wages"] += wages
        totals["worker_days"] += workers
        animals = sum("animal" in tile for _, tile in assets)
        feed_units = animals if day < last_day else 0
        if day == state.day:
            feed_units = max(
                0,
                feed_units
                - sum(tile.get("fed_today", False) for _, tile in assets if "animal" in tile),
            )
        use = min(feed_units, holdings["WHEAT"])
        holdings["WHEAT"] -= use
        buy = int(feed_units - use)
        feeding, market["WHEAT"] = quote_purchase("WHEAT", buy, market["WHEAT"], state.params)
        cash -= feeding
        totals["feed_cost"] += feeding
        minimum = min(minimum, cash)
        if cash < 0:
            feasible = False
        opponent_assets, external, _ = _daily_outputs(opponent_assets, day, cfg)
        for item, n in external.items():
            n = int(n * scenario.opponent_supply)
            _, market[item] = quote_sale(item, n, market[item], state.params)
        demand = demand_forecast(state, day, scenario)
        for item, n in demand.items():
            market[item] -= n
        assets, output, _ = _daily_outputs(assets, day, cfg, scenario.service)
        for item, n in output.items():
            fractional[item] += n
            whole = int(fractional[item])
            fractional[item] -= whole
            holdings[item] += whole
        sales = 0.0
        for item in PRODUCTS:
            # Keep next day's wheat. Its actual purchase cost is not also charged
            # as an opportunity cost: retained units reduce the next feed bill.
            keep = animals if item == "WHEAT" and day < last_day else 0
            sell = int(max(0, holdings[item] - keep))
            revenue, market[item] = quote_sale(item, sell, market[item], state.params)
            holdings[item] -= sell
            sales += revenue
        cash += sales
        totals["sales"] += sales
        rows.append(
            dict(
                day=day,
                cash=cash,
                daily_sales=sales,
                feed_purchase=feeding,
                wages=wages,
                workers=workers,
                workload=today_work,
            )
        )
    return Forecast(
        cash,
        minimum,
        totals["sales"],
        totals["feed_cost"],
        totals["wages"],
        capital,
        totals["worker_days"],
        rows,
        feasible,
        dict(
            scenario=scenario.name,
            service=scenario.service,
            hidden_opponent_inventory="not available or synthesized",
            future_opponent_investment="not modeled",
            unknown_shops="uniform expectation, range scenarios",
            timing="daily approximation; physical rollout is separate",
            asset_terminal_value=0,
            crop_replanting="not assumed",
        ),
    )


def proposals(state, allowed=None, max_assets=20):
    if len(state.animals()) + len(state.plants()) >= max_assets:
        return []
    allowed = allowed or tuple(ANIMALS) + tuple(CROPS)
    projects = []
    for item in allowed:
        sites = available_sites(state, item)
        if not sites:
            sites = available_sites(state, item, include_next_land=True)
        candidates = [Project(item, pos, land) for _, pos, land in sites]
        project = next((p for p in candidates if can_install(state, p)), None)
        if project is not None:
            projects.append(project)
    return projects


def running_reserve(state, added_animals=0, max_workers=10):
    """Two feed days and conservative two-day wage runway, in real cash."""
    animals = len(state.animals()) + added_animals
    assets = [
        (pos, tile)
        for pos, tile in state.tiles()
        if isinstance(tile, dict) and ("animal" in tile or tile.get("kind") == "PLANT")
    ]
    workers = max(1, ceil(workload(assets, state.settings) / (state.settings.turns_per_day * 0.70)))
    wages = sum(
        hire_cost(i, state.settings.hire_mult)
        for i in range(min(max(0, int(max_workers) - 1), workers - 1))
    )
    feed, _ = quote_purchase("WHEAT", animals * 2, state.market["WHEAT"], state.params)
    return feed + 2 * wages


@dataclass
class PlanForecast:
    """Inspectable daily ledger for a staged, multi-investment farm plan."""

    terminal_cash: float
    opponent_cash: float
    margin: float
    own_relative_cash: float
    minimum_cash: float
    sales: float
    opponent_sales: float
    feed_cost: float
    wages: float
    capital_cost: float
    worker_days: int
    rows: list
    feasible: bool
    assumptions: dict


def _plan_demand(state, day, scenario):
    """Closed-form equivalent of known_demand, plus uncertain future shops."""
    cfg = state.settings
    start = max(state.step, day * cfg.turns_per_day)
    end = min(cfg.last_action_step, (day + 1) * cfg.turns_per_day - 1)
    result = {item: 0.0 for item in PRODUCTS}
    if start > end:
        return result
    ticks = end // cfg.shop_interval - (start - 1) // cfg.shop_interval
    centers = end // cfg.center_interval - (start - 1) // cfg.center_interval
    for shop in state.shops:
        products = SHOPS[shop]
        for item in products:
            result[item] += ticks * (2 if len(products) == 1 else 1)
    for item in PRODUCTS:
        if item != "FERTILIZER":
            result[item] += centers
    future_unlocks = min(
        max(0, 8 - len(state.shops)),
        max(0, day // cfg.unlock_interval - state.day // cfg.unlock_interval),
    )
    if future_unlocks:
        for products in SHOPS.values():
            for item in products:
                result[item] += (
                    future_unlocks
                    * ticks
                    / len(SHOPS)
                    * (2 if len(products) == 1 else 1)
                    * scenario.unknown_demand
                )
    return result


def _plan_sales(item, own_count, rival_count, inventory, params):
    """Simultaneous unit quotes for one daily sale batch, not order prediction."""
    from .market import price

    own_cash = rival_cash = 0
    own_count, rival_count = int(own_count), int(rival_count)
    for unit in range(max(own_count, rival_count)):
        value = price(item, inventory, params)
        if value == 1:
            # Floor-priced sales do not deposit stock. Every remaining unit
            # receives the same quote, including the two sellers' shared stock.
            own_cash += max(0, own_count - unit)
            rival_cash += max(0, rival_count - unit)
            inventory += 0
            return own_cash, rival_cash, inventory
        own = unit < own_count
        rival = unit < rival_count
        own_cash += value * own
        rival_cash += value * rival
        inventory += (int(own) + int(rival)) * int(value > 1)
    return own_cash, rival_cash, inventory


@lru_cache(maxsize=512)
def _plan_geometry_work(signature, board_size):
    """Pure bounded cache: only public geometry/types, never cash or inventories."""
    from .rules import Settings

    entries = [
        (pos, {"animal": item} if item in ANIMALS else {"kind": "PLANT", "crop": item})
        for pos, item in signature
    ]
    return workload(entries, Settings(board_size=board_size))


@lru_cache(maxsize=512)
def _remaining_geometry_work(signature, positions, board_size, pickup, delivery):
    """Public remaining tasks and current positions; no private-stock cache."""
    from .rules import Settings

    if not signature:
        return 0.0
    cfg = Settings(board_size=board_size)
    remaining = [pos for pos, _ in signature]
    actions = sum(count for _, count in signature)
    if pickup:
        start = min(positions, key=lambda p: (distance(p, nearest_shed(p, cfg)), p))
        pos = nearest_shed(start, cfg)
        travel = distance(start, pos)
        actions += 1
    else:
        pos = min(positions, key=lambda p: (min(distance(p, q) for q in remaining), p))
        travel = 0
    while remaining:
        target = min(remaining, key=lambda p: (distance(pos, p), p))
        travel += distance(pos, target)
        remaining.remove(target)
        pos = target
    if delivery:
        travel += distance(pos, nearest_shed(pos, cfg))
        actions += 1
    return float(travel + actions)


def _remaining_work(entries, positions, day, final_day, settings):
    """Today's uncompleted biological jobs, without charging completed flags."""
    tasks = []
    pickup = delivery = False
    for pos, tile in entries:
        count = 0
        if "animal" in tile:
            spec = ANIMALS[tile["animal"]]
            feed = not tile["fed_today"] and day < final_day
            first = tile["placed_day"] + spec["first"]
            next_bonus = max(day + 2, first)
            next_bonus += (first - next_bonus) % spec["interval"]
            care = not tile["cared_today"] and next_bonus <= final_day
            harvest = tile["yield_units"] > 0
            fertilizer = tile["fertilizer_available"]
            count = int(feed) + int(care) + int(harvest) + int(fertilizer)
            pickup |= feed
            delivery |= harvest or fertilizer
        elif tile.get("kind") == "PLANT":
            spec = CROPS[tile["crop"]]
            age = day - tile["planted_day"]
            peak = min(spec["peak"], 10) if tile["crop"] == "MELON" else spec["peak"]
            mature = age >= spec["first"] and tile["yield_units"] > 0
            harvest = mature and (spec["interval"] or age >= peak or day == final_day)
            bonus_water = (
                not spec["interval"]
                and (spec["peak"] + 1) // 2 <= age <= spec["peak"]
                and tile["yield_units"] < spec["cap"]
            )
            water = not tile["watered_today"] and (
                (day < final_day and (not harvest or bonus_water or spec["interval"]))
                or (day == final_day and mature and bonus_water)
            )
            count = int(water) + int(harvest)
            delivery |= harvest
        if count:
            tasks.append((pos, count))
    return _remaining_geometry_work(
        tuple(sorted(tasks)), tuple(positions), settings.board_size, bool(pickup), bool(delivery)
    )


def _public_annual_rotations(state, blueprint):
    """Validate caller-observed footprints; never choose a new rival site."""
    result = {}
    size = state.settings.board_size
    for location, item, earliest in blueprint:
        if item not in CROPS or CROPS[item]["interval"]:
            continue
        pos = tuple(location)
        if len(pos) != 2 or any(type(n) is not int or not 0 <= n < size for n in pos):
            continue
        x, y = pos
        tile = state.opponent["tiles"][y][x]
        eligible = (
            tile is None
            or isinstance(tile, dict)
            and (
                tile.get("kind") == "WEED"
                or tile.get("kind") == "PLANT"
                and tile.get("crop") == item
            )
        )
        if eligible:
            result.setdefault(pos, (item, max(state.day, int(earliest))))
    return result


def _public_annual_harvest(tile, day, final_day):
    """The existing daily model's annual collection condition, before mutation."""
    item = tile.get("crop")
    if tile.get("kind") != "PLANT" or item not in CROPS or CROPS[item]["interval"]:
        return False
    spec = CROPS[item]
    age = day - tile["planted_day"]
    earned = tile["yield_units"]
    if not tile["watered_today"] and (spec["peak"] + 1) // 2 <= age <= spec["peak"]:
        earned = min(spec["cap"], earned + 1)
    peak = min(spec["peak"], 10) if item == "MELON" else spec["peak"]
    return age >= spec["first"] and earned > 0 and (age >= peak or day == final_day)


def forecast_plan(state, scenario, plan, max_workers=16, deadline=None, *, rival_rotations=None):
    """Stage purchases from real cash and account for rotations and shared prices.

    Exact transitions and live procurement remain in execution/physics. This
    ledger never installs all proposed projects at once, invents rival private
    inventory, or finances a morning purchase with its own future receipts.
    """
    from .projects import install_deadline
    from .rules import LAND_ORDER

    if deadline is not None and perf_counter() >= deadline:
        return None
    cfg = state.settings
    initial_cash = float(state.farm["money"])
    rival_initial = float(state.opponent["money"])
    if state.step > cfg.last_action_step:
        return PlanForecast(
            initial_cash,
            rival_initial,
            initial_cash - rival_initial,
            0.0,
            initial_cash,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,
            [],
            not plan.projects,
            dict(scenario=scenario.name, horizon="finished", asset_terminal_value=0),
        )
    max_workers = max(1, int(max_workers))
    service_profile = dict(getattr(plan, "service", ()))
    care_fraction = max(0.0, min(1.0, float(service_profile.get("care", 1.0))))
    fertilizer_fraction = max(0.0, min(1.0, float(service_profile.get("fertilizer", 1.0))))
    assets = [
        (pos, deepcopy(tile))
        for pos, tile in state.tiles()
        if isinstance(tile, dict) and ("animal" in tile or tile.get("kind") == "PLANT")
    ]
    rivals = [
        (pos, deepcopy(tile))
        for pos, tile in state.tiles(opponent=True)
        if isinstance(tile, dict) and ("animal" in tile or tile.get("kind") == "PLANT")
    ]
    rival_blueprint = (
        _public_annual_rotations(state, rival_rotations) if rival_rotations is not None else None
    )
    rival_active = {pos for pos, _ in rivals}
    rival_pending = {
        pos: earliest
        for pos, (_, earliest) in (rival_blueprint or {}).items()
        if pos not in rival_active
    }
    rival_weeds = {
        pos
        for pos in rival_pending
        if isinstance(state.opponent["tiles"][pos[1]][pos[0]], dict)
        and state.opponent["tiles"][pos[1]][pos[0]].get("kind") == "WEED"
    }
    grid = {(x, y): deepcopy(tile) for (x, y), tile in state.tiles()}
    rotations = {tuple(pos): item for pos, item in plan.rotations if item in CROPS}
    rotation_after = {tuple(pos): int(day) for pos, day in getattr(plan, "rotation_after", ())}
    seeds = dict(state.private["seeds"])
    inputs = {
        item: state.private["shed"].get(item, 0)
        + sum(bag.get(item, 0) for bag in state.private["inventories"])
        for item in ANIMALS
    }
    holdings = {
        item: state.private["shed"].get(item, 0)
        + sum(bag.get(item, 0) for bag in state.private["inventories"])
        for item in PRODUCTS
    }
    rival_holdings = {item: 0 for item in PRODUCTS}
    fractional = {item: 0.0 for item in PRODUCTS}
    rival_fractional = {item: 0.0 for item in PRODUCTS}
    pending = []
    seen = set()
    feasible = True
    for project in plan.projects:
        pos = tuple(project.location)
        if pos in seen or project.item not in (*ANIMALS, *CROPS):
            feasible = False
            continue
        seen.add(pos)
        tile = grid[pos]
        if isinstance(tile, dict) and (
            tile.get("animal") == project.item or tile.get("crop") == project.item
        ):
            continue
        pending.append((project, max(state.day, rotation_after.get(pos, state.day)), False))
    # A physical prefix can end after harvesting but before the next planting.
    # The rotation is still a future investment even with no live crop or
    # active Project. Carry its observation-derived cooldown into this tail.
    for pos, item in rotations.items():
        if pos in seen:
            continue
        tile = grid[pos]
        if tile is None or isinstance(tile, dict) and tile.get("kind") == "WEED":
            pending.append(
                (Project(item, pos), max(state.day, rotation_after.get(pos, state.day)), True)
            )
    cash, rival_cash = initial_cash, rival_initial
    minimum = cash
    market = dict(state.market)
    unlocked = list(state.farm["unlocked_quadrants"])
    work_cache = {}
    rows = []
    totals = dict(sales=0.0, rival_sales=0.0, feed=0.0, wages=0.0, capital=0.0, worker_days=0)
    final_day = cfg.last_action_step // cfg.turns_per_day
    exhausted = []

    def work(entries):
        key = tuple(sorted((pos, tile.get("animal") or tile.get("crop")) for pos, tile in entries))
        if key not in work_cache:
            work_cache[key] = _plan_geometry_work(key, cfg.board_size)
        return work_cache[key]

    def wage(workers, day, own=True):
        already = state.farm["hires_today"] if own else state.opponent["hires_today"]
        first = already if day == state.day else 0
        return sum(hire_cost(i, cfg.hire_mult) for i in range(first, max(first, workers - 1)))

    for day in range(state.day, final_day + 1):
        if deadline is not None and perf_counter() >= deadline:
            return None  # Incomplete forecasts can never enter the adoption score.
        turns = min(cfg.turns_per_day, cfg.last_action_step - day * cfg.turns_per_day + 1)
        if day == state.day:
            turns -= state.hour
        usable = max(1.0, turns * 0.70)
        current_day = day == state.day
        today_work = (
            _remaining_work(assets, state.positions, day, final_day, cfg)
            if current_day
            else work(assets)
        )
        can_hire_today = not current_day or state.end_of_day_step - state.step >= 4
        worker_limit = (
            max(max_workers, len(state.positions)) if can_hire_today else len(state.positions)
        )
        workers = max(1, len(state.positions) if current_day else 1, ceil(today_work / usable))
        if workers > worker_limit:
            feasible = False
        workers = min(worker_limit, workers)
        # Pay ongoing work before optional expansion; no same-day future sale
        # is available to fund this morning's investment.
        wages = wage(workers, day)
        daily_wages = wages
        cash -= wages
        totals["wages"] += wages
        setup_work = 0.0
        new_assets = []
        planned_work = today_work
        added = []
        next_pending = []
        for project, earliest, repeat in pending:
            pos = tuple(project.location)
            if earliest > day:
                next_pending.append((project, earliest, repeat))
                continue
            tile = grid[pos]
            occupied = isinstance(tile, dict) and ("animal" in tile or tile.get("kind") == "PLANT")
            if occupied:
                next_pending.append((project, day + 1, repeat))
                continue
            if day * cfg.turns_per_day + (state.hour if day == state.day else 0) > install_deadline(
                state, project
            ):
                if not repeat:
                    exhausted.append((project.item, pos))
                continue
            capital = 0.0
            needs_land = tile == "LOCKED"
            if needs_land:
                index = len(unlocked) - 1
                half = cfg.board_size // 2
                quadrant = ("N" if pos[1] < half else "S") + ("W" if pos[0] < half else "E")
                if index >= len(LAND_ORDER) or quadrant != LAND_ORDER[index]:
                    next_pending.append((project, day + 1, repeat))
                    continue
                capital += LAND_COSTS[index]
            animal = project.item in ANIMALS
            stored = inputs.get(project.item, 0) if animal else seeds.get(project.item, 0)
            if not stored:
                capital += ANIMALS[project.item]["cost"] if animal else CROPS[project.item]["seed"]
            preparation = distance(nearest_shed(pos, cfg), pos) + 2
            preparation += int(isinstance(tile, dict) and tile.get("kind") == "WEED")
            if animal:
                preparation += 2 + int(
                    not isinstance(tile, dict)
                    or tile.get("kind") != ANIMALS[project.item]["structure"]
                )
            preparation += int(needs_land)
            if turns < preparation:
                # One project's travel and preparation cannot be parallelized
                # into an instantaneous installation by hiring more workers.
                next_pending.append((project, day + 1, repeat))
                continue
            provisional = assets + [
                (
                    pos,
                    new_animal(project.item, day)
                    if animal
                    else new_plant(project.item, day, cfg.turns_per_day),
                )
            ]
            candidate_work = (
                today_work + work([*new_assets, provisional[-1]]) + setup_work + preparation
                if current_day
                else work(provisional) + setup_work + preparation
            )
            needed = max(workers, ceil(candidate_work / usable))
            if needed > worker_limit:
                next_pending.append((project, day + 1, repeat))
                continue
            extra_wages = wage(needed, day) - wage(workers, day)
            count_animals = sum("animal" in asset for _, asset in provisional)
            feed_runway, _ = quote_purchase(
                "WHEAT", count_animals * min(2, final_day - day), market["WHEAT"], state.params
            )
            next_wages = (
                sum(hire_cost(i, cfg.hire_mult) for i in range(needed - 1))
                if day < final_day
                else 0
            )
            # Reserve the feed requirement not covered by already observed grain.
            observed_grain = min(count_animals * min(2, final_day - day), holdings["WHEAT"])
            feed_credit, _ = quote_purchase("WHEAT", observed_grain, market["WHEAT"], state.params)
            reserve = max(0, feed_runway - feed_credit) + next_wages
            if cash < capital + extra_wages + reserve:
                next_pending.append((project, day + 1, repeat))
                continue
            cash -= capital + extra_wages
            totals["capital"] += capital
            totals["wages"] += extra_wages
            daily_wages += extra_wages
            workers = needed
            setup_work += preparation
            if stored:
                (inputs if animal else seeds)[project.item] -= 1
            if needs_land:
                unlocked.append(LAND_ORDER[len(unlocked) - 1])
                half = cfg.board_size // 2
                for location, old in grid.items():
                    quadrant = ("N" if location[1] < half else "S") + (
                        "W" if location[0] < half else "E"
                    )
                    if old == "LOCKED" and quadrant == unlocked[-1]:
                        grid[location] = None
            new_tile = provisional[-1][1]
            assets.append((pos, new_tile))
            new_assets.append((pos, new_tile))
            planned_work = candidate_work
            grid[pos] = new_tile
            added.append(dict(item=project.item, location=pos, repeat=repeat, capital=capital))
        pending = next_pending
        totals["worker_days"] += workers
        animals = sum("animal" in tile for _, tile in assets)
        feed_units = animals if day < final_day else 0
        if day == state.day:
            feed_units -= sum(
                tile.get("fed_today", False) for _, tile in assets if "animal" in tile
            )
        use = min(max(0, feed_units), holdings["WHEAT"])
        holdings["WHEAT"] -= use
        feed, market["WHEAT"] = quote_purchase(
            "WHEAT", int(max(0, feed_units - use)), market["WHEAT"], state.params
        )
        cash -= feed
        totals["feed"] += feed
        minimum = min(minimum, cash)
        if cash < 0:
            feasible = False
        rival_positions = [tuple(state.opponent["farmer"]), *map(tuple, state.opponent["hands"])]
        rival_work = (
            _remaining_work(rivals, rival_positions, day, final_day, cfg)
            if current_day
            else work(rivals)
        )
        rival_workers = max(
            1, len(rival_positions) if current_day else 1, ceil(rival_work / usable)
        )
        rival_limit = (
            max(max_workers, len(rival_positions)) if can_hire_today else len(rival_positions)
        )
        rival_paid_workers = min(rival_limit, rival_workers)
        rival_wages = wage(rival_paid_workers, day, own=False)
        rival_cash -= rival_wages
        rival_animals = sum("animal" in tile for _, tile in rivals)
        rival_feed_units = rival_animals if day < final_day else 0
        if day == state.day:
            rival_feed_units -= sum(
                tile.get("fed_today", False) for _, tile in rivals if "animal" in tile
            )
        rival_use = min(max(0, rival_feed_units), rival_holdings["WHEAT"])
        rival_holdings["WHEAT"] -= rival_use
        rival_feed, market["WHEAT"] = quote_purchase(
            "WHEAT", int(max(0, rival_feed_units - rival_use)), market["WHEAT"], state.params
        )
        rival_cash -= rival_feed
        rival_added = []
        rival_seed_cost = 0.0
        if rival_pending:
            new_rivals = []
            rival_setup = 0.0
            for pos, earliest in tuple(rival_pending.items()):
                if deadline is not None and perf_counter() >= deadline:
                    return None
                if earliest > day:
                    continue
                item = rival_blueprint[pos][0]
                project = Project(item, pos)
                now = day * cfg.turns_per_day + (state.hour if current_day else 0)
                if now > install_deadline(state, project):
                    del rival_pending[pos]
                    continue
                preparation = distance(nearest_shed(pos, cfg), pos) + 2 + int(pos in rival_weeds)
                if turns < preparation:
                    continue
                addition = (pos, new_plant(item, day, cfg.turns_per_day))
                candidate_work = (
                    rival_work + work([*new_rivals, addition]) + rival_setup + preparation
                    if current_day
                    else work([*rivals, addition]) + rival_setup + preparation
                )
                needed = max(rival_paid_workers, ceil(candidate_work / usable))
                if needed > rival_limit:
                    continue
                extra_wages = wage(needed, day, own=False) - wage(
                    rival_paid_workers, day, own=False
                )
                reserve_units = rival_animals * min(2, final_day - day)
                missing_feed = max(0, reserve_units - rival_holdings["WHEAT"])
                runway, _ = quote_purchase(
                    "WHEAT", int(missing_feed), market["WHEAT"], state.params
                )
                next_wages = (
                    sum(hire_cost(i, cfg.hire_mult) for i in range(needed - 1))
                    if day < final_day
                    else 0
                )
                seed_cost = CROPS[item]["seed"]
                if rival_cash < seed_cost + extra_wages + runway + next_wages:
                    continue
                rival_cash -= seed_cost + extra_wages
                rival_wages += extra_wages
                rival_seed_cost += seed_cost
                rival_paid_workers = needed
                rival_setup += preparation
                rivals.append(addition)
                new_rivals.append(addition)
                rival_weeds.discard(pos)
                del rival_pending[pos]
                rival_added.append(dict(item=item, location=pos, capital=seed_cost))
        before = {pos: tile for pos, tile in assets}
        # Recover whole observed yield collected on the first prediction day.
        # A prospective service haircut must not turn an already earned single
        # unit into 0.95 units and then lose it during integer sale conversion.
        earned_output = {item: 0 for item in PRODUCTS}
        if day == state.day:
            for _, tile in assets:
                if "animal" in tile:
                    earned_output[ANIMALS[tile["animal"]]["product"]] += tile["yield_units"]
                    earned_output["FERTILIZER"] += int(tile["fertilizer_available"])
                elif tile.get("kind") == "PLANT":
                    spec = CROPS[tile["crop"]]
                    age = day - tile["planted_day"]
                    peak = min(spec["peak"], 10) if tile["crop"] == "MELON" else spec["peak"]
                    if age >= spec["first"] and (
                        spec["interval"] or age >= peak or day == final_day
                    ):
                        earned_output[tile["crop"]] += tile["yield_units"]
        assets, output, _ = _daily_outputs(assets, day, cfg, scenario.service, care_fraction)
        for item, earned in earned_output.items():
            prospective = max(0.0, output[item] - earned * scenario.service)
            if item == "FERTILIZER":
                prospective *= fertilizer_fraction
            output[item] = earned + prospective
        rival_harvests = (
            {
                pos
                for pos, tile in rivals
                if pos in rival_blueprint and _public_annual_harvest(tile, day, final_day)
            }
            if rival_blueprint
            else set()
        )
        rivals, external, _ = _daily_outputs(rivals, day, cfg)
        if rival_harvests and day < final_day:
            active_rivals = {pos for pos, _ in rivals}
            for pos in rival_harvests - active_rivals:
                rival_pending[pos] = max(day + 1, rival_blueprint[pos][1])
        active_positions = {pos for pos, _ in assets}
        for pos in before:
            if pos not in active_positions:
                grid[pos] = None
                if pos in rotations and day < final_day:
                    pending.append(
                        (
                            Project(rotations[pos], pos),
                            max(day + 1, rotation_after.get(pos, day + 1)),
                            True,
                        )
                    )
        for pos, tile in assets:
            grid[pos] = tile
        for item in PRODUCTS:
            fractional[item] += output[item]
            own_whole = int(fractional[item])
            fractional[item] -= own_whole
            holdings[item] += own_whole
            rival_fractional[item] += external[item] * scenario.opponent_supply
            rival_whole = int(rival_fractional[item])
            rival_fractional[item] -= rival_whole
            rival_holdings[item] += rival_whole
        demand = _plan_demand(state, day, scenario)
        for item in PRODUCTS:
            market[item] -= demand[item]
        keep = 2 * animals if day < final_day else 0
        # Four daily deposit/sale waves are an explicit throughput approximation.
        waves = max(1, turns // 6)
        stock_peak = sum(inputs.values()) + keep + ceil(sum(output.values()) / waves)
        if stock_peak > cfg.shed_capacity:
            feasible = False
        sales = rival_sales = 0.0
        for item in PRODUCTS:
            own_keep = min(holdings[item], keep) if item == "WHEAT" else 0
            rival_keep = (
                min(rival_holdings[item], rival_animals)
                if item == "WHEAT" and day < final_day
                else 0
            )
            own_count = int(max(0, holdings[item] - own_keep))
            rival_count = int(max(0, rival_holdings[item] - rival_keep))
            revenue, rival_revenue, market[item] = _plan_sales(
                item, own_count, rival_count, market[item], state.params
            )
            holdings[item] -= own_count
            rival_holdings[item] -= rival_count
            sales += revenue
            rival_sales += rival_revenue
        cash += sales
        rival_cash += rival_sales
        totals["sales"] += sales
        totals["rival_sales"] += rival_sales
        row = dict(
            day=day,
            cash=cash,
            opponent_cash=rival_cash,
            margin=cash - rival_cash,
            daily_sales=sales,
            opponent_sales=rival_sales,
            feed_purchase=feed,
            wages=daily_wages,
            workers=workers,
            workload=planned_work,
            stock_peak=stock_peak,
            installed=added,
            production=dict(output),
            opponent_production=dict(external),
        )
        if rival_blueprint is not None:
            row.update(
                opponent_installed=rival_added,
                opponent_seed_cost=rival_seed_cost,
                opponent_wages=rival_wages,
                opponent_feed_purchase=rival_feed,
                opponent_workers=rival_paid_workers,
            )
        rows.append(row)
    unfinished = [
        (project.item, tuple(project.location)) for project, _, repeat in pending if not repeat
    ]
    feasible = feasible and not exhausted and not unfinished
    return PlanForecast(
        cash,
        rival_cash,
        cash - rival_cash,
        cash - initial_cash,
        minimum,
        totals["sales"],
        totals["rival_sales"],
        totals["feed"],
        totals["wages"],
        totals["capital"],
        totals["worker_days"],
        rows,
        feasible,
        dict(
            scenario=scenario.name,
            family=plan.family,
            service=scenario.service,
            hidden_opponent_inventory="unknown; initial private stock is not synthesized",
            future_opponent_investment=(
                "existing observed annual footprints only; cash-funded repeat; no expansion"
                if rival_blueprint is not None
                else "not modeled; public output scaled by scenario"
            ),
            unknown_shops="uniform expectation with scenario range",
            timing="daily purchases before own receipts; simultaneous daily sale batches",
            maintenance="ideal calendar with monetizable-output haircut",
            measured_service=dict(care=care_fraction, fertilizer=fertilizer_fraction),
            care_banking="observed care preserved; future bank increments use measured fraction",
            fertilizer_collection="observed yield preserved; new output uses measured fraction",
            opponent_service="unknown profile; ideal care/collection, scenario-scaled supply",
            transport="nearest-neighbor work plus four daily deposit/sale waves",
            crop_replanting="next day after removal; seed and preparation charged every cycle",
            purchases="observed inputs allocated by quantity; land cost once per unlocked quadrant",
            asset_terminal_value=0,
            unliquidated_inventory_value=0,
            unfinished_projects=unfinished,
            expired_projects=exhausted,
        ),
    )
