"""Own private state only. Observations are copied, never modified or guessed."""

from copy import deepcopy
from dataclasses import dataclass

from .rules import MARKET_PARAMS, Settings

_SCALAR_TYPES = {type(None), bool, int, float, complex, str, bytes}


def _structural_copy(value, memo):
    """Copy the scalar-heavy observation tree, preserving shared references."""
    kind = type(value)
    if kind in _SCALAR_TYPES:
        return value
    identity = id(value)
    if identity in memo:
        return memo[identity]
    if kind is dict:
        result = {}
        memo[identity] = result
        for key, item in value.items():
            if type(key) not in _SCALAR_TYPES:
                key = deepcopy(key, memo)
            result[key] = item if type(item) in _SCALAR_TYPES else _structural_copy(item, memo)
        return result
    if kind is list:
        result = value.copy()
        memo[identity] = result
        for index, item in enumerate(value):
            if type(item) not in _SCALAR_TYPES:
                result[index] = _structural_copy(item, memo)
        return result
    # Tuples, container subclasses and unknown nested values
    # retain Python's deepcopy semantics instead of assuming their contents.
    return deepcopy(value, memo)


@dataclass
class State:
    step: int
    player: int
    farm: dict
    private: dict
    opponent: dict
    market: dict
    shops: list
    settings: Settings
    params: dict

    @classmethod
    def from_observation(cls, observation, configuration=None):
        obs = observation
        settings = Settings.from_config(configuration)
        player = int(obs.get("player", 0))
        params = deepcopy(MARKET_PARAMS)
        patches = (configuration or {}).get("marketParams", {})
        for item, patch in patches.items():
            if item in params:
                params[item].update(patch)
        # Explicit market params, when provided, take precedence.
        for item, patch in obs["market"].get("params", {}).items():
            if item in params:
                params[item].update(patch)
        farms = obs["farms"]
        return cls(
            int(obs.get("step", 0)),
            player,
            deepcopy(farms[player]),
            deepcopy(obs["private"]),
            deepcopy(farms[1 - player]),
            deepcopy(obs["market"]["inventory"]),
            list(obs.get("town", {}).get("unlocked_shops", [])),
            settings,
            params,
        )

    def clone(self):
        if type(self) is not State:
            return deepcopy(self)
        result = State.__new__(State)
        memo = {id(self): result}
        # Settings contains only immutable rule values. Unknown settings types
        # still go through deepcopy; all mutable observation fields are copied.
        if type(self.settings) is Settings:
            memo[id(self.settings)] = self.settings
        result.__dict__ = _structural_copy(self.__dict__, memo)
        return result

    @property
    def day(self):
        return self.step // self.settings.turns_per_day

    @property
    def hour(self):
        return self.step % self.settings.turns_per_day

    @property
    def positions(self):
        return [tuple(self.farm["farmer"]), *map(tuple, self.farm["hands"])]

    @property
    def end_of_day_step(self):
        return min((self.day + 1) * self.settings.turns_per_day - 1, self.settings.last_action_step)

    def tiles(self, opponent=False):
        farm = self.opponent if opponent else self.farm
        for y, row in enumerate(farm["tiles"]):
            for x, tile in enumerate(row):
                yield (x, y), tile

    def animals(self):
        return [
            (pos, tile) for pos, tile in self.tiles() if isinstance(tile, dict) and "animal" in tile
        ]

    def plants(self):
        return [
            (pos, tile)
            for pos, tile in self.tiles()
            if isinstance(tile, dict) and tile.get("kind") == "PLANT"
        ]

    def shed_used(self):
        return sum(self.private["shed"].values())


def distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def nearest_shed(pos, settings):
    if (
        type(settings) is Settings
        and type(settings.board_size) is int
        and type(pos) in (tuple, list)
        and len(pos) >= 2
        and type(pos[0]) is int
        and type(pos[1]) is int
    ):
        # The four sheds form {h-1, h} x {h-1, h}. For integer coordinates,
        # Manhattan distance has a unique closest choice on each axis.
        h = settings.board_size // 2
        return h - 1 if pos[0] < h else h, h - 1 if pos[1] < h else h
    return min(settings.shed_tiles(), key=lambda p: (distance(pos, p), p))


def move_toward(pos, target):
    if pos[0] != target[0]:
        return ["EAST" if pos[0] < target[0] else "WEST"]
    if pos[1] != target[1]:
        return ["SOUTH" if pos[1] < target[1] else "NORTH"]
    return ["PASS"]
