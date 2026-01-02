"""Set operations for polygons (Intersection, Union, Difference)."""

import jax
import jax.numpy as jnp
from geometry import Polygon
from core import _intersection, _clean_vertices, _get_clipped_segments, _stitch_segments


@jax.jit(static_argnames=["max_vertices"])
def intersection(
    polygon1: Polygon, polygon2: Polygon, max_vertices: int = 256
) -> Polygon:
    """Computes the intersection of two polygons using the Sutherland-Hodgman algorithm.

    Assumptions:
        - polygon2 is convex and vertices are in counter-clockwise order.
        - max_vertices is sufficient to hold the result.

    args:
        polygon1: Subject polygon.
        polygon2: Clip polygon (Must be CONVEX and CCW).
        max_vertices: Size of the output buffer.

    Returns:
        The intersection polygon.

    Warning:
        If the result requires more than `max_vertices`, the polygon will be truncated
        and the `overflow` flag will be set to `True`. Check `result.overflow`
        and retry with a larger `max_vertices` if necessary.
    """
    vertices, count = _intersection(polygon1.vertices, polygon2.vertices, max_vertices)

    # Check for overflow:
    # 1. Output overflow (count reached limit)
    # 2. Input overflow (inputs were truncated before processing).
    #    Note: _intersection iterates polygon2 fully (if valid JAX loop), but clamps polygon1.
    input_ovf = polygon1.count > max_vertices
    res_ovf = count >= max_vertices

    # If count reached the limit, we assume overflow/truncation risk
    overflow = input_ovf | res_ovf

    # Clean the output to remove duplicates
    vertices, count = _clean_vertices(vertices, count)

    safe_count = jnp.minimum(count, max_vertices)

    return Polygon(vertices=vertices, count=safe_count, overflow=overflow)


@jax.jit(static_argnames=["max_vertices"])
def union(polygon1: Polygon, polygon2: Polygon, max_vertices: int = 256) -> Polygon:
    """Computes the union of two polygons (P1 U P2).

    Approximation:
    Since JAX requires static shapes and strict topology is hard to maintain, we use a
    Clipping-based approach to stitch the outer boundary segments.
    Result correctness relies on the assumption that the union forms a single connected loop.

    Args:
        polygon1: First polygon.
        polygon2: Second polygon.
        max_vertices: Size of the output buffer.

    Returns:
        The union polygon.

    Warning:
        If the result requires more than `max_vertices`, the polygon will be truncated
        and the `overflow` flag will be set to `True`. Check `result.overflow`
        and retry with a larger `max_vertices` if necessary.
    """
    # Robust implementation using segment stitching
    # 1. Collect segments of A outside B
    # 2. Collect segments of B outside A
    # 3. Stitch

    # Use max_vertices for capacity estimation
    seg1, count1 = _get_clipped_segments(
        polygon1.vertices,
        polygon1.count,
        polygon2.vertices,
        polygon2.count,
        max_vertices,
        keep_inside=False,
    )

    seg2, count2 = _get_clipped_segments(
        polygon2.vertices,
        polygon2.count,
        polygon1.vertices,
        polygon1.count,
        max_vertices,
        keep_inside=False,
    )

    # Pad to safe size (using 2x for safety as in _get_clipped_segments)
    capacity = max_vertices * 2

    # We need to concatenate seg1 (valid 0..count1) and seg2 (valid 0..count2)
    # Using jnp.concatenate on raw seg1/seg2 is WRONG because seg1 has padding at end which would separate valid seg1 from valid seg2.
    # We use dynamic_update_slice to pack them.

    all_segments = jnp.zeros((capacity * 2, 2, 2))
    all_segments = jax.lax.dynamic_update_slice(all_segments, seg1, (0, 0, 0))
    all_segments = jax.lax.dynamic_update_slice(all_segments, seg2, (count1, 0, 0))

    all_count = count1 + count2

    # Stitch
    vertices, count = _stitch_segments(all_segments, all_count, max_vertices)

    # Input overflow logic
    # _get_clipped_segments uses max_vertices for capacity.
    # If inputs are huge, it might truncate.
    input_ovf = (polygon1.count > max_vertices) | (polygon2.count > max_vertices)
    res_ovf = count >= max_vertices
    overflow = input_ovf | res_ovf

    # Clean the output to remove duplicates
    vertices, count = _clean_vertices(vertices, count)

    safe_count = jnp.minimum(count, max_vertices)

    return Polygon(vertices=vertices, count=safe_count, overflow=overflow)


@jax.jit(static_argnames=["max_vertices"])
def difference(
    polygon1: Polygon, polygon2: Polygon, max_vertices: int = 256
) -> Polygon:
    """Computes the difference P1 - P2.

    Args:
        polygon1: Subject polygon.
        polygon2: Clip polygon.
        max_vertices: Size of the output buffer.

    Returns:
        The difference polygon.

    Warning:
        If the result requires more than `max_vertices`, the polygon will be truncated
        and the `overflow` flag will be set to `True`. Check `result.overflow`
        and retry with a larger `max_vertices` if necessary.
    """
    # Difference = Parts of P1 OUTSIDE P2 + Parts of P2 INSIDE P1 (reversed)
    # Why P2 inside P1 reversed? Imagine P1 is big square, P2 is hole.
    # Boundary follows P1, then jumps to P2 hole (reversed).

    seg1, count1 = _get_clipped_segments(
        polygon1.vertices,
        polygon1.count,
        polygon2.vertices,
        polygon2.count,
        max_vertices,
        keep_inside=False,
    )

    seg2, count2 = _get_clipped_segments(
        polygon2.vertices,
        polygon2.count,
        polygon1.vertices,
        polygon1.count,
        max_vertices,
        keep_inside=True,
    )

    # Reverse segments for the "hole" part (P2 inside P1)
    # Segment is (p1, p2). Reversing means (p2, p1).
    seg2 = seg2[:, ::-1, :]

    # Pack segments
    capacity = max_vertices * 2
    all_segments = jnp.zeros((capacity * 2, 2, 2))
    all_segments = jax.lax.dynamic_update_slice(all_segments, seg1, (0, 0, 0))
    all_segments = jax.lax.dynamic_update_slice(all_segments, seg2, (count1, 0, 0))

    all_count = count1 + count2

    vertices, count = _stitch_segments(all_segments, all_count, max_vertices)

    input_ovf = (polygon1.count > max_vertices) | (polygon2.count > max_vertices)
    res_ovf = count >= max_vertices
    overflow = input_ovf | res_ovf

    # Clean the output to remove duplicates
    vertices, count = _clean_vertices(vertices, count)

    safe_count = jnp.minimum(count, max_vertices)

    return Polygon(vertices=vertices, count=safe_count, overflow=overflow)
