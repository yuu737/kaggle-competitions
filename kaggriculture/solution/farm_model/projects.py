"""Investment intent, compatible sites and conservative cash-return deadlines."""

from dataclasses import dataclass

from .rules import ANIMALS, CROPS, LAND_ORDER
from .state import distance, nearest_shed


@dataclass(frozen=True)
class Project:
    item: str
    location: tuple
    needs_land: bool = False

    @property
    def animal(self):
        return self.item in ANIMALS


def installed(state, project):
    x, y = project.location
    tile = state.farm["tiles"][y][x]
    return isinstance(tile, dict) and (
        tile.get("animal") == project.item or tile.get("crop") == project.item
    )


def install_deadline(state, project):
    """Latest install action leaving time for first collection and final sale.

    Crops require same-day watering. Animals can first earn fertilizer on the
    following day, before their main product matures. Reserve travel from the
    farmer's daily reset position and a physical delivery to the nearest shed.
    This is a necessary timing guard, not a promise of profitable investment.
    """
    cfg = state.settings
    wait = 1 if project.animal else CROPS[project.item]["first"]
    pos = project.location
    collection_travel = distance(cfg.shed_tiles()[0], pos)
    delivery = distance(pos, nearest_shed(pos, cfg)) + 1
    latest_day = (cfg.last_action_step - collection_travel - delivery) // cfg.turns_per_day - wait
    return (latest_day + 1) * cfg.turns_per_day - (1 if project.animal else 2)


def _site_details(state, item, pos, include_next_land):
    x, y = pos
    tile = state.farm["tiles"][y][x]
    next_quadrant = None
    index = len(state.farm["unlocked_quadrants"]) - 1
    if include_next_land and index < len(LAND_ORDER):
        next_quadrant = LAND_ORDER[index]
    half = state.settings.board_size // 2
    quadrant = ("N" if y < half else "S") + ("W" if x < half else "E")
    land = tile == "LOCKED" and quadrant == next_quadrant
    weed = isinstance(tile, dict) and tile.get("kind") == "WEED"
    barn = (
        item in ANIMALS
        and isinstance(tile, dict)
        and tile.get("kind") == ANIMALS[item]["structure"]
        and "animal" not in tile
    )
    if tile is None or weed or barn or land:
        return land, int(weed) + int(item in ANIMALS and not barn)
    return None


def available_sites(state, item, include_next_land=False):
    """Owned compatible empty structures are reusable; occupied assets are not."""
    sites = []
    for pos, _ in state.tiles():
        details = _site_details(state, item, pos, include_next_land)
        if details is not None:
            land, preparation = details
            cost = distance(pos, nearest_shed(pos, state.settings)) + preparation
            sites.append((cost, pos, land))
    return sorted(sites)


def can_install(state, project):
    """Reject incompatible sites and setups too late to produce cashable output."""
    if installed(state, project):
        return False
    details = _site_details(state, project.item, project.location, True)
    if details is None or (details[0] and not project.needs_land):
        return False
    x, y = project.location
    tile = state.farm["tiles"][y][x]
    weed = isinstance(tile, dict) and tile.get("kind") == "WEED"
    deadline = install_deadline(state, project)
    if project.animal:
        barn = isinstance(tile, dict) and tile.get("kind") == ANIMALS[project.item]["structure"]
        stages = 1 + int(weed) + int(not barn)
        routes = []
        for unit, pos in enumerate(state.positions):
            bag = state.private["inventories"][unit]
            if bag.get(project.item, 0):
                routes.append(distance(pos, project.location) + stages)
            else:
                shed = nearest_shed(pos, state.settings)
                wait = int(state.private["shed"].get(project.item, 0) == 0)
                routes.append(
                    wait + distance(pos, shed) + 1 + distance(shed, project.location) + stages
                )
    else:
        wait = int(state.private["seeds"].get(project.item, 0) == 0)
        routes = [wait + distance(pos, project.location) + int(weed) + 1 for pos in state.positions]
    return state.step + min(routes) - 1 <= deadline


def pending_projects(state):
    """Reuse observed inputs only while installation can still return cash.

    Expired material remains in authoritative inventory. Its sunk cost is not
    a reason to displace harvest, delivery or maintenance.
    """
    assigned = set()
    projects = []
    for item in (*ANIMALS, *CROPS):
        count = (
            state.private["seeds"].get(item, 0)
            if item in CROPS
            else state.private["shed"].get(item, 0)
            + sum(bag.get(item, 0) for bag in state.private["inventories"])
        )
        for _ in range(count):
            candidates = [
                Project(item, pos, land)
                for _, pos, land in available_sites(state, item)
                if pos not in assigned
            ]
            project = next((p for p in candidates if can_install(state, p)), None)
            if project is None:
                break
            projects.append(project)
            assigned.add(project.location)
    return projects


def input_counts(state):
    """Observed setup inputs, counted once across the shed and all workers.

    This is a quantity ledger, not a prediction of orders being filled. Animal
    products and wheat held as feed never count as livestock setup inputs.
    """
    return {
        item: (
            state.private["seeds"].get(item, 0)
            if item in CROPS
            else state.private["shed"].get(item, 0)
            + sum(bag.get(item, 0) for bag in state.private["inventories"])
        )
        for item in (*ANIMALS, *CROPS)
    }


def unique_projects(state, projects, blocked_sites=()):
    """Keep the first feasible contract for each site, preserving plan order."""
    blocked = {tuple(pos) for pos in blocked_sites}
    seen = set()
    result = []
    for project in projects:
        if (
            project.location in blocked
            or project.location in seen
            or not can_install(state, project)
        ):
            continue
        seen.add(project.location)
        result.append(project)
    return tuple(result)


def reconcile_projects(state, committed=(), blocked_sites=()):
    """Preserve viable intents and attach only unclaimed *observed* materials.

    An intent can still need a purchase. Reserving an input for it decreases
    the observed quantity ledger by at most one; it never creates inventory.
    Any spare observed input can be assigned to another feasible owned site.
    Completed, incompatible, and expired intents cannot consume a reservation.
    """
    blocked = {tuple(pos) for pos in blocked_sites}
    projects = list(unique_projects(state, committed, blocked))
    assigned = {project.location for project in projects}
    available = input_counts(state)
    for project in projects:
        if available[project.item] > 0:
            available[project.item] -= 1
    for item in (*ANIMALS, *CROPS):
        for _ in range(available[item]):
            candidates = (
                Project(item, pos, land)
                for _, pos, land in available_sites(state, item)
                if pos not in assigned and pos not in blocked
            )
            project = next((p for p in candidates if can_install(state, p)), None)
            if project is None:
                break
            projects.append(project)
            assigned.add(project.location)
    return tuple(projects)
