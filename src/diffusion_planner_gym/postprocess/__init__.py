"""Post-processing applied to the planner trajectory before the ego follows it.

Add a new post-processor as a module here and register it in `POSTPROCESSORS`.
"""

from __future__ import annotations

from typing import Any

from .base import PostProcessor
from .casadi_optimization import CasadiOptimization
from .none import NoPostProcess

POSTPROCESSORS: dict[str, type[PostProcessor]] = {
    "none": NoPostProcess,
    "casadi-optimization": CasadiOptimization,
}


def make_postprocessor(
    postprocess: str | PostProcessor, **kwargs: Any
) -> PostProcessor:
    if not isinstance(postprocess, str):
        return postprocess
    if postprocess not in POSTPROCESSORS:
        raise ValueError(
            f"Unknown postprocess {postprocess!r}; choose from {sorted(POSTPROCESSORS)}"
        )
    return POSTPROCESSORS[postprocess](**kwargs)


__all__ = [
    "POSTPROCESSORS",
    "CasadiOptimization",
    "NoPostProcess",
    "PostProcessor",
    "make_postprocessor",
]
