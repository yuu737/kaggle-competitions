"""Exact price and lockstep execution; future opposing orders are hypotheses."""

import math
from functools import lru_cache

from .rules import ANIMALS, CROPS, LAND_COSTS, LAND_ORDER, PRODUCTS, SHOPS, hire_cost


def shape(name, amount, throughput):
    x = max(0.0, amount)
    if name == "sq":
        return x * x
    if name == "sqrt":
        return math.sqrt(x)
    if name == "log":
        return math.log(1.0 + x)
    if name == "log10":
        return math.log10(1.0 + x)
    if name == "hinge" and throughput > 0:
        u = x / throughput
        return u + 8.0 * max(0.0, u - 1.0) ** 2
    return x


@lru_cache(maxsize=16384, typed=True)
def _curve_price(inventory, base, baseline, throughput, name, target, below):
    """Exact unit quote, keyed only by immutable active-curve values."""
    amplitude = target * base / shape(name, throughput, throughput)
    delta = amplitude * shape(name, abs(inventory - baseline), throughput)
    return max(1, int(round(base + (delta if below else -delta))))


_NUMERIC_TYPES = (int, float, bool)


def price(item, inventory, params):
    p = params[item]
    baseline, base, throughput = p["I0"], p["base"], p["T"]
    below = inventory < baseline
    name = p["below_func"] if below else p["above_func"]
    target = p["below_target"] if below else p["above_target"]
    if (
        type(inventory) in _NUMERIC_TYPES
        and type(base) in _NUMERIC_TYPES
        and type(baseline) in _NUMERIC_TYPES
        and type(throughput) in _NUMERIC_TYPES
        and type(target) in _NUMERIC_TYPES
        and type(name) is str
    ):
        return _curve_price(inventory, base, baseline, throughput, name, target, below)
    # Unknown parameter objects keep the original arithmetic and are never
    # retained in a cache whose value could become stale after mutation.
    return _curve_price.__wrapped__(inventory, base, baseline, throughput, name, target, below)


def quote_sale(item, count, inventory, params):
    """Return cash and resulting stock, including the $1 non-deposit rule."""
    cash = 0
    count = max(0, int(count))
    for index in range(count):
        unit = price(item, inventory, params)
        if unit == 1:
            # Stock stays fixed at the price floor, so the remaining quotes
            # are identical. This is the official non-deposit rule.
            return cash + count - index, inventory
        cash += unit
        inventory += 1
    return cash, inventory


def quote_purchase(item, count, inventory, params):
    cash = 0
    for _ in range(max(0, int(count))):
        inventory -= 1
        cash += price(item, inventory, params)
    return cash, inventory


def spawn_hand(farm, settings):
    positions = [tuple(farm["farmer"]), *map(tuple, farm["hands"])]
    tiles = settings.shed_tiles()
    pos = min(tiles, key=lambda p: (positions.count(p), tiles.index(p)))
    return list(pos)


def buy_land(farm, settings):
    index = len(farm["unlocked_quadrants"]) - 1
    if index >= len(LAND_COSTS) or farm["money"] < LAND_COSTS[index]:
        return False
    farm["money"] -= LAND_COSTS[index]
    quadrant = LAND_ORDER[index]
    farm["unlocked_quadrants"].append(quadrant)
    half = settings.board_size // 2
    for y, row in enumerate(farm["tiles"]):
        for x, tile in enumerate(row):
            q = ("N" if y < half else "S") + ("W" if x < half else "E")
            if tile == "LOCKED" and q == quadrant:
                row[x] = None
    return True


def parse_order(order):
    if not isinstance(order, list) or not order:
        return None
    op = order[0]
    if op in ("HIRE", "BUY_LAND"):
        return [op, None, 1]
    if op not in ("SELL", "BUY_PRODUCT", "BUY_SEED", "BUY_ANIMAL") or len(order) < 3:
        return None
    try:
        n = int(order[2])
    except (ValueError, TypeError):
        return None
    return [op, order[1], n] if n > 0 else None


def execute_market(accounts, inventory, queues, settings, params):
    """Mutate isolated accounts; quote both players before committing either.

    accounts are in *seat order*, not own/opponent order. An empty/invalid slot
    still occupies its original position. Returns successful unit/atomic counts.
    """
    trace = []
    queues = [q[: settings.max_orders] for q in queues]
    for index in range(max(map(len, queues), default=0)):
        orders = [parse_order(q[index]) if index < len(q) else None for q in queues]
        counts = [0, 0]
        for seat, order in enumerate(orders):
            if order is None:
                continue
            farm, private = accounts[seat]["farm"], accounts[seat]["private"]
            if order[0] == "HIRE":
                cost = hire_cost(farm["hires_today"], settings.hire_mult)
                if farm["money"] >= cost:
                    farm["money"] -= cost
                    farm["hires_today"] += 1
                    farm["hands"].append(spawn_hand(farm, settings))
                    private["inventories"].append({})
                    counts[seat] = 1
                orders[seat] = None
            elif order[0] == "BUY_LAND":
                counts[seat] = int(buy_land(farm, settings))
                orders[seat] = None
        while any(order is not None and order[2] > 0 for order in orders):
            quotes = [None, None]
            for seat, order in enumerate(orders):
                if order is None or order[2] <= 0:
                    continue
                op, item, _ = order
                if op == "SELL" and item in PRODUCTS:
                    quotes[seat] = price(item, inventory[item], params)
                elif op == "BUY_PRODUCT" and item in ("WHEAT", "FERTILIZER"):
                    quotes[seat] = price(item, inventory[item] - 1, params)
                elif op == "BUY_SEED" and item in CROPS:
                    quotes[seat] = CROPS[item]["seed"]
                elif op == "BUY_ANIMAL" and item in ANIMALS:
                    quotes[seat] = ANIMALS[item]["cost"]
                else:
                    orders[seat] = None
            committed = False
            for seat, unit_price in enumerate(quotes):
                if unit_price is None:
                    continue
                op, item, _ = orders[seat]
                farm, private = accounts[seat]["farm"], accounts[seat]["private"]
                shed = private["shed"]
                if op == "SELL":
                    ok = shed.get(item, 0) > 0
                    if ok:
                        shed[item] -= 1
                        farm["money"] += unit_price
                        inventory[item] += int(unit_price > 1)
                else:
                    ok = farm["money"] >= unit_price
                    if op != "BUY_SEED":
                        ok = ok and sum(shed.values()) < settings.shed_capacity
                    if ok:
                        farm["money"] -= unit_price
                        bag = private["seeds"] if op == "BUY_SEED" else shed
                        bag[item] = bag.get(item, 0) + 1
                        if op == "BUY_PRODUCT":
                            inventory[item] -= 1
                if ok:
                    orders[seat][2] -= 1
                    counts[seat] += 1
                    committed = True
                else:
                    orders[seat] = None
            if not committed:
                break
        for seat, n in enumerate(counts):
            if index < len(queues[seat]):
                trace.append(dict(seat=seat, index=index, order=queues[seat][index], executed=n))
    return trace


def consume_town(inventory, shops, step, settings):
    if step % settings.shop_interval == 0:
        for shop in shops:
            products = SHOPS[shop]
            for item in products:
                inventory[item] -= 2 if len(products) == 1 else 1
    if step % settings.center_interval == 0:
        for item in PRODUCTS:
            if item != "FERTILIZER":
                inventory[item] -= 1
