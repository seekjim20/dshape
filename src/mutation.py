"""Mutation operations for polygons (Buffer, Offset)."""

import jax
import jax.numpy as jnp
from geometry import Polygon
from core import _buffer, _clean_vertices


@jax.jit(static_argnames=["max_vertices"])
def buffer(polygon: Polygon, distance: float, max_vertices: int = 256) -> Polygon:
    """Computes the buffer of a polygon.

    Args:
        polygon: Input polygon.
        distance: Buffer distance (positive for dilation, negative for erosion).
        max_vertices: Size of output buffer.

    Returns:
        Buffered polygon.

    Warning:
        If the result requires more than `max_vertices`, the polygon will be truncated
        and the `overflow` flag will be set to `True`. Check `result.overflow`
        and retry with a larger `max_vertices` if necessary.
    """
    vertices, count = _buffer(polygon.vertices, polygon.count, distance, max_vertices)

    # Note: _buffer usually cleans vertices, so checking input count > max_vertices
    # might be too aggressive if cleaning reduces it significantly.
    # But usually buffer INCREASES count. So if input > output cap, it's risky.
    # _buffer implementation handles input iteration.

    # Keep output overflow check primarily.
    overflow = count >= max_vertices

    # Clean the output to remove duplicates
    vertices, count = _clean_vertices(vertices, count)

    safe_count = jnp.minimum(count, max_vertices)

    return Polygon(vertices=vertices, count=safe_count, overflow=overflow)


def offset(polygon: Polygon, dx: float, dy: float) -> Polygon:
    """Translates the polygon by (dx, dy).

    Args:
        polygon: Input polygon.
        dx: X translation.
        dy: Y translation.

    Returns:
        Offset polygon.
    """
    return Polygon(
        vertices=polygon.vertices + jnp.array([dx, dy]),
        count=polygon.count,
        overflow=polygon.overflow,
    )
