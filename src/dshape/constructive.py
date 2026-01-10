"""Constructive geometry operations (Convex Hull)."""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from . import geometry
from . import core


@jax.jit(static_argnames=["max_vertices"])
def convex_hull(
    polygon: geometry.Polygon, max_vertices: int = None
) -> geometry.Polygon:
    """Computes the convex hull of the polygon.

    Args:
        polygon: Input polygon (can be multi-ring).
        max_vertices: Size of output buffer. If None, uses polygon.max_vertices.

    Returns:
        Polygon representing the convex hull.
    """
    if max_vertices is None:
        max_vertices = polygon.vertices.shape[0]

    hull_verts, hull_count = core._convex_hull(
        polygon.vertices, polygon.count, max_vertices
    )

    ring_counts = jnp.array([hull_count], dtype=jnp.int32)
    overflow = hull_count >= max_vertices

    return geometry.Polygon(
        vertices=hull_verts,
        count=hull_count,
        ring_counts=ring_counts,
        overflow=overflow,
    )
