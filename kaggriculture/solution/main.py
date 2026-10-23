"""Clean entry point. Keep agent the last callable for the official loader."""

from farm_model import controller as _controller
from farm_model import execution as _execution

_ENGINE = _controller.Controller(
    _execution.Policy(
        farm_planning=True,
        service_tours=True,
        large_batches=True,
        cash_bridge=True,
        rival_cycles=False,
        rotation_only=True,
        preserve_rotations=True,
        additive_growth=True,
        max_workers=16,
        max_assets=96,
        candidate_limit=4,
        physical_candidates=1,
        search_seconds=0.27,
    )
)


def agent(observation, configuration=None):
    return _ENGINE.decide(observation, configuration)
