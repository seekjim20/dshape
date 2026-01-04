from .geometry import Polygon
from .set_ops import intersection, union, difference
from .mutation import buffer, offset, rotate, scale
from .constructive import convex_hull

# -- Monkey Patching Methods to Polygon --


def _add(self, other):
    """Union operator (+)"""
    if not isinstance(other, Polygon):
        return NotImplemented
    return union([self, other])


def _sub(self, other):
    """Difference operator (-)"""
    if not isinstance(other, Polygon):
        return NotImplemented
    return difference(self, other)


def _mul(self, other):
    """Intersection operator (*)"""
    if not isinstance(other, Polygon):
        return NotImplemented
    return intersection(self, other)


Polygon.__add__ = _add
Polygon.__sub__ = _sub
Polygon.__mul__ = _mul

Polygon.convex_hull = property(convex_hull)
Polygon.buffer = buffer
Polygon.offset = offset
Polygon.rotate = rotate
Polygon.scale = scale

__all__ = [
    "Polygon",
    "intersection",
    "union",
    "difference",
    "buffer",
    "offset",
    "rotate",
    "scale",
    "convex_hull",
]
