from . import geometry
from . import set_ops
from . import mutation
from . import constructive
from . import plotting

# -- Monkey Patching Methods to Polygon --


def _add(self, other):
    """Union operator (+)"""
    if not isinstance(other, geometry.Polygon):
        return NotImplemented
    return set_ops.union([self, other])


def _sub(self, other):
    """Difference operator (-)"""
    if not isinstance(other, geometry.Polygon):
        return NotImplemented
    return set_ops.difference(self, other)


def _mul(self, other):
    """Intersection operator (*)"""
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


def _contains(self, point):
    """Checks if the polygon contains the given point."""
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


def _line_segment_intersection_method(self, other):
    """Intersection with another LineSegment or Polygon."""
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
