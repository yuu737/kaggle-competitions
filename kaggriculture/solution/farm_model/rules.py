"""Rules pinned to kaggle-environments 1.32.7; hypotheses live elsewhere."""

from dataclasses import dataclass
from functools import lru_cache

CROPS = {
    "WHEAT": dict(seed=10, first=2, peak=4, interval=0, cap=6),
    "CARROT": dict(seed=20, first=2, peak=3, interval=0, cap=4),
    "TOMATO": dict(seed=50, first=8, peak=8, interval=1, cap=4),
    "STRAWBERRY": dict(seed=100, first=10, peak=10, interval=2, cap=4),
    "MELON": dict(seed=80, first=10, peak=12, interval=0, cap=6),
}
ANIMALS = {
    "GOOSE": dict(cost=300, structure="COOP", first=4, interval=1, cap=4, product="EGG"),
    "COW": dict(cost=400, structure="PASTURE", first=8, interval=2, cap=6, product="MILK"),
    "SHEEP": dict(cost=500, structure="PASTURE", first=6, interval=3, cap=6, product="WOOL"),
}
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
# base, throughput, scarcity shape/target, surplus shape/target
_CURVES = {
    "WHEAT": (25, 400, "sqrt", 0.80, "log", 0.20),
    "CARROT": (35, 450, "hinge", 1.0, "sqrt", 0.70),
    "TOMATO": (60, 200, "hinge", 0.40, "sqrt", 0.60),
    "STRAWBERRY": (120, 100, "sqrt", 0.70, "linear", 1.60),
    "MELON": (250, 300, "log", 0.20, "sq", 3.60),
    "EGG": (50, 332, "hinge", 0.40, "log", 0.20),
    "MILK": (160, 122, "sqrt", 0.60, "linear", 1.60),
    "WOOL": (200, 105, "log", 0.20, "sq", 3.20),
    "FERTILIZER": (100, 200, "linear", 0.40, "linear", 0.40),
}
MARKET_PARAMS = {
    item: dict(
        base=b, I0=10000, T=t, below_func=bf, below_target=bt, above_func=af, above_target=at
    )
    for item, (b, t, bf, bt, af, at) in _CURVES.items()
}
SHOPS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}
MOVES = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}
LAND_ORDER = ("NE", "SW", "SE")
LAND_COSTS = (1000, 2000, 4000)


@lru_cache(maxsize=16)
def _shed_tiles(board_size):
    h = board_size // 2
    return ((h - 1, h - 1), (h, h - 1), (h - 1, h), (h, h))


@dataclass(frozen=True)
class Settings:
    turns_per_day: int = 24
    episode_steps: int = 720
    board_size: int = 10
    shed_capacity: int = 100
    max_orders: int = 10
    hire_mult: int = 1
    shop_interval: int = 4
    center_interval: int = 24
    unlock_interval: int = 3

    @classmethod
    def from_config(cls, configuration=None):
        cfg = configuration or {}
        names = {
            "turns_per_day": "turnsPerDay",
            "episode_steps": "episodeSteps",
            "board_size": "boardSize",
            "shed_capacity": "shedCapacity",
            "max_orders": "maxMarketOrdersPerTurn",
            "hire_mult": "farmHandCostMult",
            "shop_interval": "townShopSellInterval",
            "center_interval": "townCenterSellInterval",
            "unlock_interval": "townShopUnlockInterval",
        }
        defaults = cls()
        return cls(
            **{
                name: max(
                    0 if name == "hire_mult" else 1, int(cfg.get(key, getattr(defaults, name)))
                )
                for name, key in names.items()
            }
        )

    @property
    def last_action_step(self):
        # Official interpreter ends after action at episodeSteps - 2.
        return self.episode_steps - 2

    def shed_tiles(self):
        # Board geometry is immutable and independent of the other settings.
        return _shed_tiles(self.board_size)


def hire_cost(already, multiplier=1):
    a, b = 1, 1
    for _ in range(already):
        a, b = b, a + b
    return a * multiplier


def new_plant(crop, day, turns):
    spec = CROPS[crop]
    return dict(
        kind="PLANT",
        crop=crop,
        planted_day=day,
        watered_today=False,
        consecutive_unwatered=1,
        yield_units=0 if spec["interval"] else 1,
        max_lifespan_step=-1 if spec["interval"] else (day + spec["peak"] + 1) * turns,
        fertilized_until_day=-1,
    )


def new_animal(animal, day):
    return dict(
        kind=ANIMALS[animal]["structure"],
        animal=animal,
        placed_day=day,
        yield_units=0,
        consecutive_unfed=0,
        fed_today=False,
        cared_today=False,
        fertilizer_available=False,
        pending_care_bonus=0,
    )
