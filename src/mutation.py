"""Mutation operations for polygons (Buffer, Offset)."""

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
import geometry
import core


@jax.jit(static_argnames=["max_vertices"])
def buffer(
    polygon: geometry.Polygon, distance: ArrayLike, max_vertices: int = 256
) -> geometry.Polygon:
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
    vertices, count = core._buffer(
        polygon.vertices, polygon.count, distance, max_vertices
    )

    # Note: _buffer usually cleans vertices, so checking input count > max_vertices
    # might be too aggressive if cleaning reduces it significantly.
    # But usually buffer INCREASES count. So if input > output cap, it's risky.
    # _buffer implementation handles input iteration.

    # Keep output overflow check primarily.
    overflow = count >= max_vertices

    # Clean the output to remove duplicates
    vertices, count = core._clean_vertices(vertices, count)

    safe_count = jnp.minimum(count, max_vertices)

    return geometry.Polygon(vertices=vertices, count=safe_count, overflow=overflow)


def offset(polygon: geometry.Polygon, dx: ArrayLike, dy: ArrayLike) -> geometry.Polygon:
    """Translates the polygon by (dx, dy).

    Args:
        polygon: Input polygon.
        dx: X translation.
        dy: Y translation.

    Returns:
        Offset polygon.
    """
    return geometry.Polygon(
        vertices=polygon.vertices + jnp.array([dx, dy]),
        count=polygon.count,
        overflow=polygon.overflow,
    )
