"""dshape: Differentiable Shape Operations with JAX.

This module exposes the main API for the dshape library.
It performs monkey patching to add convenient methods (like .buffer(), .contains())
and operator overloads (+, -, *) to the Polygon and LineSegment classes.
"""

from . import geometry
from . import set_ops
from . import mutation
from . import constructive
from . import plotting

# type hints
from jax import Array
from typing import Union, Any


__all__ = [
    "geometry.Polygon",
    "geometry.LineSegment",
    "geometry.Point",
    "set_ops.intersection",
    "set_ops.union",
    "set_ops.difference",
    "set_ops.line_segment_intersection",
    "set_ops.line_segment_polygon_intersection",
    "mutation.buffer",
    "mutation.translate",
    "mutation.rotate",
    "mutation.scale",
    "constructive.convex_hull",
]
