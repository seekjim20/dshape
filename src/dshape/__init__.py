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


# -- Monkey Patching Methods to Polygon --


def _add(self: geometry.Polygon, other: Any) -> geometry.Polygon:
    """Union operator (+).

    Args:
        other: Another Polygon.

    Returns:
        The union of the two polygons.
    """
    if not isinstance(other, geometry.Polygon):
        return NotImplemented
    return set_ops.union([self, other])


def _sub(self: geometry.Polygon, other: Any) -> geometry.Polygon:
    """Difference operator (-).

    Args:
        other: Another Polygon to subtract.

    Returns:
        The difference polygon (self - other).
    """
    if not isinstance(other, geometry.Polygon):
        return NotImplemented
    return set_ops.difference(self, other)


def _mul(self: geometry.Polygon, other: Any) -> geometry.Polygon:
    """Intersection operator (*).

    Args:
        other: Another Polygon to intersect with.

    Returns:
        The intersection of the two polygons.
    """
    if not isinstance(other, geometry.Polygon):
        return NotImplemented
    return set_ops.intersection(self, other)


geometry.Polygon.__add__ = _add
geometry.Polygon.__sub__ = _sub
geometry.Polygon.__mul__ = _mul

geometry.Polygon.convex_hull = property(constructive.convex_hull)
geometry.Polygon.buffer = mutation.buffer
geometry.Polygon.translate = mutation.translate
geometry.Polygon.rotate = mutation.rotate
geometry.Polygon.scale = mutation.scale


def _contains(self: geometry.Polygon, point: Union[geometry.Point, Array]) -> bool:
    """Checks if the polygon contains the given point.

    Args:
        point: A dshape.geometry.Point or an array-like (x, y).

    Returns:
        True if the point is strictly inside the polygon's exterior rings
        and outside its interior rings (holes), False otherwise.
    """
    # Ensure point is a Point object or array-like
    if hasattr(point, "xy"):
        pt_arr = point.xy
    else:
        # Assuming array-like
        from jax import numpy as jnp

        pt_arr = jnp.asarray(point)

    from . import core

    return core._is_point_in_polygon(
        pt_arr, self.vertices, self.count, self.ring_counts
    )


geometry.Polygon.contains = _contains


# -- Monkey Patching Methods to LineSegment --


def _line_segment_intersection_method(
    self: geometry.LineSegment, other: Union[geometry.LineSegment, geometry.Polygon]
) -> Union[geometry.Point, list[geometry.LineSegment], None]:
    """Computes the intersection with another geometric object.

    Args:
        other: A LineSegment or a Polygon.

    Returns:
        - If other is a LineSegment: returns a Point (single intersection) or None.
        - If other is a Polygon: returns a list of LineSegments (parts inside polygon).

    Raises:
        TypeError: If 'other' is not a supported type.
    """
    if isinstance(other, geometry.LineSegment):
        return set_ops.line_segment_intersection(self, other)
    elif isinstance(other, geometry.Polygon):
        return set_ops.line_segment_polygon_intersection(self, other)
    else:
        raise TypeError(f"Cannot compute intersection with {type(other).__name__}")


geometry.LineSegment.intersection = _line_segment_intersection_method

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
