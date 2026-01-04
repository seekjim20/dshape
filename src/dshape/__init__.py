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
geometry.Polygon.offset = mutation.offset
geometry.Polygon.rotate = mutation.rotate
geometry.Polygon.scale = mutation.scale

__all__ = [
    "geometry.Polygon",
    "set_ops.intersection",
    "set_ops.union",
    "set_ops.difference",
    "mutation.buffer",
    "mutation.offset",
    "mutation.rotate",
    "mutation.scale",
    "constructive.convex_hull",
]
