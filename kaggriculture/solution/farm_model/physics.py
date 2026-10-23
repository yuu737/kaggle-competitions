"""Deterministic own-farm transitions, differential-tested against official code.

No hidden opponent inventory and no future RNG are synthesized here. Future
shop/weed/supply events must be supplied explicitly by a forecast scenario.
"""

from .market import consume_town, execute_market
from .rules import ANIMALS, CROPS, MOVES, new_animal, new_plant


def take(bag, item, count=1):
    if bag.get(item, 0) < count:
        return False
    bag[item] -= count
    if not bag[item]:
        del bag[item]
    return True


def apply_unit(state, unit, action):
    """Return (effect, discarded_units). A missing effect denotes a no-op."""
    if unit >= len(state.positions) or not isinstance(action, list) or not action:
        return None, 0
    op = action[0]
    farm, private, cfg = state.farm, state.private, state.settings
    x, y = state.positions[unit]
    while len(private["inventories"]) <= unit:
        private["inventories"].append({})
    bag = private["inventories"][unit]
    if op in MOVES:
        dx, dy = MOVES[op]
        nx, ny = x + dx, y + dy
        if 0 <= nx < cfg.board_size and 0 <= ny < cfg.board_size:
            if unit == 0:
                farm["farmer"] = [nx, ny]
            else:
                farm["hands"][unit - 1] = [nx, ny]
            return "MOVE", 0
        return None, 0
    tile = farm["tiles"][y][x]
    adjacent = (x, y) in cfg.shed_tiles()
    shed = private["shed"]
    if op == "DROP" and adjacent:
        lost = 0
        n_total = sum(bag.values())
        for item, count in list(bag.items()):
            put = min(count, max(0, cfg.shed_capacity - sum(shed.values())))
            shed[item] = shed.get(item, 0) + put
            lost += count - put
            del bag[item]
        return ("DROP" if n_total else None), lost
    if op == "PICKUP" and adjacent and len(action) >= 2:
        item = action[1]
        count = min(int(action[2]) if len(action) > 2 else 1, shed.get(item, 0))
        if count > 0:
            shed[item] -= count
            bag[item] = bag.get(item, 0) + count
            return "PICKUP", 0
        return None, 0
    if op == "PLACE" and len(action) >= 2:
        item = action[1]
        if (
            item in ANIMALS
            and isinstance(tile, dict)
            and tile.get("kind") == ANIMALS[item]["structure"]
            and "animal" not in tile
        ):
            if take(bag, item):
                farm["tiles"][y][x] = new_animal(item, state.day)
                return "PLACE_ANIMAL", 0
            return None, 0
        if adjacent:
            count = min(
                int(action[2]) if len(action) > 2 else 1,
                bag.get(item, 0),
                max(0, cfg.shed_capacity - sum(shed.values())),
            )
            if count > 0:
                take(bag, item, count)
                shed[item] = shed.get(item, 0) + count
                return "PLACE_SHED", 0
        return None, 0
    if tile == "LOCKED":
        return None, 0
    if op == "PLANT" and len(action) >= 2:
        crop = action[1]
        if tile is None and crop in CROPS and private["seeds"].get(crop, 0) > 0:
            private["seeds"][crop] -= 1
            farm["tiles"][y][x] = new_plant(crop, state.day, cfg.turns_per_day)
            return "PLANT", 0
    elif op == "WATER" and isinstance(tile, dict) and tile.get("kind") == "PLANT":
        if not tile["watered_today"]:
            tile["watered_today"] = True
            spec = CROPS[tile["crop"]]
            age = state.day - tile["planted_day"]
            if not spec["interval"] and (spec["peak"] + 1) // 2 <= age <= spec["peak"]:
                bonus = 2 if tile["fertilized_until_day"] >= state.day else 1
                tile["yield_units"] = min(spec["cap"], tile["yield_units"] + bonus)
            return "WATER", 0
    elif op == "HARVEST" and isinstance(tile, dict) and tile.get("yield_units", 0) > 0:
        if tile.get("kind") == "PLANT":
            if state.day - tile["planted_day"] < CROPS[tile["crop"]]["first"]:
                return None, 0
            product = tile["crop"]
            if not CROPS[product]["interval"]:
                farm["tiles"][y][x] = None
        elif "animal" in tile:
            product = ANIMALS[tile["animal"]]["product"]
        else:
            return None, 0
        bag[product] = bag.get(product, 0) + tile["yield_units"]
        tile["yield_units"] = 0
        return "HARVEST", 0
    elif op == "FERTILIZE" and isinstance(tile, dict) and tile.get("kind") == "PLANT":
        if take(bag, "FERTILIZER"):
            tile["fertilized_until_day"] = max(tile.get("fertilized_until_day", -1), state.day + 2)
            return "FERTILIZE", 0
    elif op == "DIG" and tile is not None and not (isinstance(tile, dict) and "animal" in tile):
        farm["tiles"][y][x] = None
        return "DIG", 0
    elif op in ("BUILD_COOP", "BUILD_PASTURE") and tile is None:
        farm["tiles"][y][x] = {"kind": "COOP" if op == "BUILD_COOP" else "PASTURE"}
        return "BUILD", 0
    elif isinstance(tile, dict) and "animal" in tile:
        if op == "FEED" and not tile["fed_today"] and take(bag, "WHEAT"):
            tile["fed_today"] = True
            return "FEED", 0
        if op == "CARE" and not tile["cared_today"]:
            tile["cared_today"] = True
            return "CARE", 0
        if op == "COLLECT_FERTILIZER" and tile["fertilizer_available"]:
            tile["fertilizer_available"] = False
            bag["FERTILIZER"] = bag.get("FERTILIZER", 0) + 1
            return "COLLECT_FERTILIZER", 0
    return None, 0


def apply_workers(state, actions):
    demand = {}
    for action in actions:
        if isinstance(action, list) and len(action) >= 2 and action[0] == "PLANT":
            demand[action[1]] = demand.get(action[1], 0) + 1
    blocked = {crop for crop, n in demand.items() if n > state.private["seeds"].get(crop, 0)}
    trace = []
    for unit, action in enumerate(actions[: len(state.positions)]):
        if (
            isinstance(action, list)
            and len(action) >= 2
            and action[0] == "PLANT"
            and action[1] in blocked
        ):
            trace.append(dict(unit=unit, action=action, effect=None, discarded=0))
            continue
        effect, lost = apply_unit(state, unit, action)
        trace.append(dict(unit=unit, action=action, effect=effect, discarded=lost))
    return trace


def refresh_tile(tile, day, settings):
    """Daily biological transition, including official care-banking order."""
    if not isinstance(tile, dict):
        return tile
    if tile.get("kind") == "PLANT":
        watered = tile["watered_today"]
        tile["consecutive_unwatered"] = 0 if watered else tile["consecutive_unwatered"] + 1
        tile["watered_today"] = False
        if tile["consecutive_unwatered"] >= 2:
            return {"kind": "WEED"}
        spec = CROPS[tile["crop"]]
        delta = day + 1 - tile["planted_day"] - spec["first"]
        if spec["interval"] and delta >= 0 and delta % spec["interval"] == 0:
            count = delta // spec["interval"] + 1
            if count <= spec["cap"]:
                bonus = int(watered and tile.get("fertilized_until_day", -1) >= day)
                tile["yield_units"] = min(spec["cap"], tile["yield_units"] + 1 + bonus)
                if count == spec["cap"]:
                    tile["max_lifespan_step"] = (day + 2) * settings.turns_per_day
    elif "animal" in tile:
        fed, cared = tile["fed_today"], tile["cared_today"]
        tile["consecutive_unfed"] = 0 if fed else tile["consecutive_unfed"] + 1
        spec = ANIMALS[tile["animal"]]
        if tile["consecutive_unfed"] >= 2:
            return {"kind": spec["structure"]}
        delta = day + 1 - tile["placed_day"] - spec["first"]
        if delta >= 0 and delta % spec["interval"] == 0:
            bonus = tile.get("pending_care_bonus", 0) if fed else 0
            tile["yield_units"] = min(spec["cap"], tile["yield_units"] + 1 + bonus)
            tile["pending_care_bonus"] = 0
        if fed and cared:
            tile["pending_care_bonus"] = tile.get("pending_care_bonus", 0) + 1
        tile["fertilizer_available"] = True
        tile["fed_today"] = tile["cared_today"] = False
    return tile


def decay(state):
    for (x, y), tile in state.tiles():
        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            life = tile["max_lifespan_step"]
            if life >= 0 and state.step >= life and (state.step - life) % 2 == 0:
                tile["yield_units"] -= 1
                if tile["yield_units"] <= 0:
                    state.farm["tiles"][y][x] = {"kind": "WEED"}


def advance(state, action, external_sales=None, new_shops=(), weeds=()):
    """Mutate a branch. Exogenous events are explicit approximate assumptions."""
    workers = apply_workers(state, [action.get("farmer", ["PASS"]), *action.get("hands", [])])
    # Only own account has private holdings. The other account has no orders.
    own = dict(farm=state.farm, private=state.private)
    dummy = dict(farm=state.opponent, private=dict(shed={}, seeds={}, inventories=[]))
    accounts = [own, dummy] if state.player == 0 else [dummy, own]
    queues = [action.get("market", []), []] if state.player == 0 else [[], action.get("market", [])]
    market = execute_market(accounts, state.market, queues, state.settings, state.params)
    for item, count in (external_sales or {}).items():
        # Background injection is not an exact prediction of simultaneous orders.
        for _ in range(max(0, int(count))):
            from .market import price

            state.market[item] += int(price(item, state.market[item], state.params) > 1)
    consume_town(state.market, state.shops, state.step, state.settings)
    decay(state)
    lost = sum(entry["discarded"] for entry in workers)
    if (state.step + 1) % state.settings.turns_per_day == 0:
        for (x, y), tile in list(state.tiles()):
            state.farm["tiles"][y][x] = refresh_tile(tile, state.day, state.settings)
        for x, y in weeds:
            if state.farm["tiles"][y][x] is None:
                state.farm["tiles"][y][x] = {"kind": "WEED"}
        for bag in state.private["inventories"]:
            for item, n in list(bag.items()):
                put = min(n, max(0, state.settings.shed_capacity - state.shed_used()))
                state.private["shed"][item] = state.private["shed"].get(item, 0) + put
                lost += n - put
                del bag[item]
        state.farm["farmer"] = list(state.settings.shed_tiles()[0])
        state.farm["hands"] = []
        state.farm["hires_today"] = 0
        state.private["inventories"] = [{}]
        state.shops.extend(new_shops)
    state.step += 1
    return dict(workers=workers, market=market, discarded=lost)
