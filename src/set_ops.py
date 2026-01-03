"""Set operations for polygons (Intersection, Union, Difference)."""

import jax
import jax.numpy as jnp
import geometry
import core
from typing import Sequence


@jax.jit(static_argnames=["max_vertices"])
def intersection(
    polygon1: geometry.Polygon, polygon2: geometry.Polygon, max_vertices: int = 256
) -> geometry.Polygon:
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
    vertices, count = core._intersection(
        polygon1.vertices, polygon2.vertices, max_vertices
    )

    # Check for overflow:
    # 1. Output overflow (count reached limit)
    # 2. Input overflow (inputs were truncated before processing).
    #    Note: _intersection iterates polygon2 fully (if valid JAX loop), but clamps polygon1.
    input_ovf = polygon1.count > max_vertices
    res_ovf = count >= max_vertices

    # If count reached the limit, we assume overflow/truncation risk
    overflow = input_ovf | res_ovf

    # Clean the output to remove duplicates
    vertices, count = core._clean_vertices(vertices, count)

    safe_count = jnp.minimum(count, max_vertices)

    return geometry.Polygon(vertices=vertices, count=safe_count, overflow=overflow)


@jax.jit(static_argnames=["max_vertices"])
def union(
    polygons: Sequence[geometry.Polygon], max_vertices: int = 256
) -> geometry.Polygon:
    """Computes the union of multiple polygons.

    Args:
        polygons: List or sequence of polygons.
        max_vertices: Size of the output buffer.

    Returns:
        The union polygon.

    Warning:
        If the result requires more than `max_vertices`, the polygon will be truncated
        and the `overflow` flag will be set to `True`. Check `result.overflow`
        and retry with a larger `max_vertices` if necessary.
    """
    # Robust implementation using segment stitching

    num_polys = len(polygons)
    if num_polys == 0:
        return geometry.Polygon(vertices=jnp.zeros((0, 2)), count=0)

    # We maintain a large buffer for all resulting segments before stitching.
    # Capacity estimation: num_polys * max_vertices * 2?
    # If list is long, this might be huge.
    # Assuming small list for now (e.g. 5).
    # Ideally should be dynamic or static bounded.

    # Let's cap total segments at max_vertices * 4 for now to keep JIT sanity
    TOTAL_CAPACITY = max_vertices * 4

    all_segments = jnp.zeros((TOTAL_CAPACITY, 2, 2))
    all_count = 0
    all_overflow = False

    # Helper to extract edges from a polygon
    def extract_edges(poly):
        N = poly.vertices.shape[0]
        c = poly.count

        # We can reuse _insert_intersections if we treat P2 as empty?
        # Or just manually expand.
        # Let's write a simple expansion.

        idxs = jnp.arange(max_vertices)  # Assume poly vertices <= max_vertices

        def get_seg(i):
            idx1 = i
            idx2 = jnp.where(i + 1 == c, 0, i + 1)
            p1 = poly.vertices[idx1]
            p2 = poly.vertices[idx2]
            valid = i < c
            return jnp.stack([p1, p2]), valid

        segs, valids = jax.vmap(get_seg)(idxs)

        # Pack
        # We need a fixed size buffer for segments.
        # Let's use max_vertices capacity.
        return segs, c

    # Iterate over each polygon as the "Subject"
    for i in range(num_polys):
        subject = polygons[i]

        # Get initial segments
        # Note: We assume subject has <= max_vertices edges.
        current_segments, current_count = extract_edges(subject)

        # Iterate over other polygons to clip against
        for j in range(num_polys):
            if i == j:
                continue

            clipper = polygons[j]

            # Clip current_segments against clipper
            # Keep OUTSIDE parts
            # Capacity for intermediate clip: max_vertices * 2 (standard heuristic)
            current_segments, current_count = core._clip_segments(
                current_segments,
                current_count,
                clipper.vertices,
                clipper.count,
                max_vertices * 2,
                keep_inside=False,
            )

        # Add to main accumulator
        # We cannot use dynamic slice on source with dynamic count.
        # We use a scan loop to copy valid segments.

        # We want to copy current_segments[0..current_count] to all_segments[all_count..]

        def copy_step(state, k):
            buf, base_ptr = state
            # segment to copy
            seg = current_segments[k]
            # conditions
            src_valid = k < current_count
            dst_idx = base_ptr + k
            dst_valid = dst_idx < TOTAL_CAPACITY

            should_copy = src_valid & dst_valid

            buf = buf.at[dst_idx].set(jnp.where(should_copy, seg, buf[dst_idx]))
            return (buf, base_ptr), None

        # Scan over all potential segments in current_segments
        # shape is (max_vertices * 2, 2, 2)
        src_len = current_segments.shape[0]
        (all_segments, _), _ = jax.lax.scan(
            copy_step, (all_segments, all_count), jnp.arange(src_len)
        )

        # Clamp current_count to fit
        remaining = TOTAL_CAPACITY - all_count
        to_add = jnp.minimum(current_count, remaining)

        # If current_count > remaining -> local overflow
        local_ovf = current_count > remaining
        all_overflow = all_overflow | local_ovf

        all_count = all_count + to_add
        all_overflow = all_overflow | subject.overflow  # Propagate input overflow

    # Remove duplicates (coincident edges from overlapping boundaries)
    all_segments, all_count = core._remove_duplicate_segments(all_segments, all_count)

    # Stitch
    vertices, final_count = core._stitch_segments(all_segments, all_count, max_vertices)

    res_ovf = final_count >= max_vertices
    overflow = all_overflow | res_ovf

    # Clean
    vertices, final_count = core._clean_vertices(vertices, final_count)

    safe_count = jnp.minimum(final_count, max_vertices)

    return geometry.Polygon(vertices=vertices, count=safe_count, overflow=overflow)


@jax.jit(static_argnames=["max_vertices"])
def difference(
    polygon1: geometry.Polygon, polygon2: geometry.Polygon, max_vertices: int = 256
) -> geometry.Polygon:
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

    seg1, count1 = core._get_clipped_segments(
        polygon1.vertices,
        polygon1.count,
        polygon2.vertices,
        polygon2.count,
        max_vertices,
        keep_inside=False,
    )

    seg2, count2 = core._get_clipped_segments(
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

    vertices, count = core._stitch_segments(all_segments, all_count, max_vertices)

    input_ovf = (polygon1.count > max_vertices) | (polygon2.count > max_vertices)
    res_ovf = count >= max_vertices
    overflow = input_ovf | res_ovf

    # Clean the output to remove duplicates
    vertices, count = core._clean_vertices(vertices, count)

    safe_count = jnp.minimum(count, max_vertices)

    return geometry.Polygon(vertices=vertices, count=safe_count, overflow=overflow)
