"""Constructive geometry operations (Convex Hull)."""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import geometry
import core


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

    # Flatten all vertices just in case, though Polygon vertices are already flat (N, 2).
    # But we should respect `count` to ignore garbage.
    # The `ring_counts` are ignored; hull works on point cloud of all rings.

    hull_verts, hull_count = core._convex_hull(
        polygon.vertices, polygon.count, max_vertices
    )

    # Convex Hull is always a single ring
    ring_counts = jnp.array([hull_count], dtype=jnp.int32)

    # Check for overflow?
    # Core doesn't return overflow boolean explicitly yet, but clamps.
    # If hull_count > max_vertices (impossible by definition if max_vertices >= input count),
    # but theoretically if input is huge and max_vertices is small.
    overflow = hull_count >= max_vertices
    # Strictly if it equals max_vertices it might be full, but usually overflow means "exceeded".
    # _convex_hull clamps count.

    return geometry.Polygon(
        vertices=hull_verts,
        count=hull_count,
        ring_counts=ring_counts,
        overflow=overflow,
    )
