"""Set operations for polygons (Intersection, Union, Difference)."""

import jax
import jax.numpy as jnp
from . import geometry
from . import core
from typing import Sequence


@jax.jit(static_argnames=["max_vertices", "max_rings"])
def intersection(
    polygon1: geometry.Polygon,
    polygon2: geometry.Polygon,
    max_vertices: int = 256,
    max_rings: int = 16,
) -> geometry.Polygon:
    """Computes the intersection of two polygons using the Sutherland-Hodgman algorithm.

    Assumptions:
        - polygon2 is convex and vertices are in counter-clockwise order.
        - max_vertices is sufficient to hold the result.

    args:
        polygon1: Subject polygon.
        polygon2: Clip polygon (Must be CONVEX and CCW).
        max_vertices: Size of the output buffer.
        max_rings: Maximum number of output rings.

    Returns:
        The intersection polygon.
    """
    # Handle multi-ring P1.
    p1_rings = polygon1.ring_counts

    # We need to map over P1 rings
    # Check shape of p1_rings
    N_RINGS_IN = p1_rings.shape[0]
    p1_starts = jnp.cumsum(jnp.pad(p1_rings, (1, 0))[:-1])

    def process_ring(i):
        # Extract ring i
        start = p1_starts[i]
        c = p1_rings[i]

        # Valid if c > 0

        # Construct a buffer for this ring
        idxs = jnp.arange(max_vertices)
        read_idxs = start + idxs
        read_idxs = jnp.minimum(read_idxs, polygon1.vertices.shape[0] - 1)

        ring_verts = polygon1.vertices[read_idxs]

        # Call intersection (assumes single ring output from single ring input + convex clip)
        clipped_verts, clipped_count = core._intersection(
            ring_verts, c, polygon2.vertices, polygon2.count, max_vertices
        )

        # If ring was invalid, clipped_count should be 0?
        # If c=0, _intersection should produce 0 count if logic holds.
        # But let's mask it.
        clipped_count = jnp.where(c > 0, clipped_count, 0)

        return clipped_verts, clipped_count

    # Iterate over all possible rings of P1
    out_verts_stack, out_counts_stack = jax.vmap(process_ring)(jnp.arange(N_RINGS_IN))

    # Pack into (max_vertices, 2) and (max_rings,)
    final_verts = jnp.zeros((max_vertices, 2))
    final_rings = jnp.zeros(max_rings, dtype=jnp.int32)

    init_state = (final_verts, final_rings, 0, 0)  # buf, rings, v_ptr, r_ptr

    def pack_ring(state, i):
        buf, rings, v_ptr, r_ptr = state

        cnt = out_counts_stack[i]
        verts = out_verts_stack[i]

        # Valid if P1 had this ring and intersection produced vertices
        is_valid = (i < N_RINGS_IN) & (p1_rings[i] > 0) & (cnt > 0)

        # Check capacity
        has_space_v = (v_ptr + cnt) <= max_vertices
        has_space_r = r_ptr < max_rings

        do_write = is_valid & has_space_v & has_space_r

        def copy_v(s, k):
            b, p = s
            v = verts[k]
            should_copy = do_write & (k < cnt)
            b = b.at[p].set(jnp.where(should_copy, v, b[p]))
            p = p + jnp.where(should_copy, 1, 0)
            return (b, p), None

        (buf, v_ptr), _ = jax.lax.scan(copy_v, (buf, v_ptr), jnp.arange(max_vertices))

        rings = rings.at[r_ptr].set(jnp.where(do_write, cnt, rings[r_ptr]))
        r_ptr = r_ptr + jnp.where(do_write, 1, 0)

        return (buf, rings, v_ptr, r_ptr), None

    (final_verts, final_rings, final_total_count, _), _ = jax.lax.scan(
        pack_ring, init_state, jnp.arange(N_RINGS_IN)
    )

    input_ovf = polygon1.count > max_vertices
    total_generated = jnp.sum(out_counts_stack)
    res_ovf = total_generated > max_vertices
    overflow = input_ovf | res_ovf

    return geometry.Polygon(
        vertices=final_verts,
        count=final_total_count,
        ring_counts=final_rings,
        overflow=overflow,
    )


@jax.jit(static_argnames=["max_vertices", "max_rings"])
def union(
    polygons: Sequence[geometry.Polygon], max_vertices: int = 256, max_rings: int = 16
) -> geometry.Polygon:
    """Computes the union of multiple polygons.

    Args:
        polygons: List or sequence of polygons.
        max_vertices: Size of the output buffer.
        max_rings: Maximum number of output rings.

    Returns:
        The union polygon.
    """
    # Robust implementation using segment stitching

    num_polys = len(polygons)
    if num_polys == 0:
        return geometry.Polygon(
            vertices=jnp.zeros((0, 2)), count=0, ring_counts=jnp.zeros(1)
        )

    TOTAL_CAPACITY = max_vertices * 4

    all_segments = jnp.zeros((TOTAL_CAPACITY, 2, 2))
    all_count = 0
    all_overflow = False

    def extract_edges(poly):
        N = poly.vertices.shape[0]
        c = poly.count
        rings = poly.ring_counts
        n_rings = rings.shape[0]
        starts = jnp.cumsum(jnp.pad(rings, (1, 0))[:-1])

        idxs = jnp.arange(max_vertices)  # Max vertices of P

        # Determine next index
        def get_seg(i):
            # Find ring
            # Using new helper from core or re-implementing inline?
            # It's better to use core helper if possible, but it's not exposed in __init__ maybe?
            # It's in core. Let's replicate logic or assume core exposed it.
            # _get_ring_aware_next_index is in core.
            idx_next = core._get_ring_aware_next_index(i, starts, rings, n_rings)

            p1 = poly.vertices[i]
            p2 = poly.vertices[idx_next]

            # Simple validity check: i < c
            # (And implicit valid ring structure)

            # Find which ring i is in to check 'local < rc' if we want strictly
            is_after = i >= starts
            k = jnp.sum(is_after) - 1
            k = jnp.maximum(0, jnp.minimum(k, n_rings - 1))
            rc = rings[k]
            s = starts[k]
            local = i - s

            valid = (i < c) & (local < rc) & (rc > 0)

            return jnp.stack([p1, p2]), valid

        segs, valids = jax.vmap(get_seg)(idxs)
        return segs, valids

    # Iterate over each polygon as the "Subject"
    for i in range(num_polys):
        subject = polygons[i]

        current_segments_raw, current_valids = extract_edges(subject)

        # Compacting
        temp_buf = jnp.zeros_like(current_segments_raw)

        def pack_s(state, k):
            b, p = state
            v = current_segments_raw[k]
            do = current_valids[k]
            b = b.at[p].set(jnp.where(do, v, b[p]))
            p = p + jnp.where(do, 1, 0)
            return (b, p), None

        (current_segments_compact, current_count), _ = jax.lax.scan(
            pack_s, (temp_buf, 0), jnp.arange(max_vertices)
        )

        current_segments = current_segments_compact

        # Iterate over other polygons to clip against
        # GLOBAL CLIPPING: Subject must be OUTSIDE all other polygons.
        # But for Union, is it (A - B) + (B - A) + (A & B)?
        # Union = (A outside B) U (B outside A) ... wait.
        # Union(A,B) = Parts of A outside B + Parts of B outside A?
        # No.
        # If A & B overlap, we want the "merged" shell.
        # The boundary of Union(A, B) consists of:
        # - Edges of A that are OUTSIDE B
        # - Edges of B that are OUTSIDE A
        # Yes.

        # So for each polygon i, we keep edges that are OUTSIDE all polygons j != i.

        for j in range(num_polys):
            if i == j:
                continue

            clipper = polygons[j]

            # Clip against ENTIRE clipper polygon
            # keep_inside = False (Keep Outside)

            new_segs, new_cnt = core._clip_segments_multiring(
                current_segments,
                current_count,
                clipper.vertices,
                clipper.count,
                clipper.ring_counts,
                max_vertices * 2,
                keep_inside=False,
            )

            current_segments = new_segs
            current_count = new_cnt

        # Add accumulation
        def copy_step(state, k):
            buf, base_ptr = state
            seg = current_segments[k]
            src_valid = k < current_count
            dst_idx = base_ptr + k
            dst_valid = dst_idx < TOTAL_CAPACITY
            should_copy = src_valid & dst_valid
            buf = buf.at[dst_idx].set(jnp.where(should_copy, seg, buf[dst_idx]))
            return (buf, base_ptr), None

        src_len = current_segments.shape[0]
        (all_segments, _), _ = jax.lax.scan(
            copy_step, (all_segments, all_count), jnp.arange(src_len)
        )
        remaining = TOTAL_CAPACITY - all_count
        to_add = jnp.minimum(current_count, remaining)
        local_ovf = current_count > remaining
        all_overflow = all_overflow | local_ovf
        all_count = all_count + to_add
        all_overflow = all_overflow | subject.overflow

    all_segments, all_count = core._remove_duplicate_segments(all_segments, all_count)
    vertices, final_count, final_rings = core._stitch_segments(
        all_segments, all_count, max_vertices, max_rings
    )
    res_ovf = final_count >= max_vertices
    overflow = all_overflow | res_ovf
    vertices, final_count = core._clean_vertices(vertices, final_count)
    safe_count = jnp.minimum(final_count, max_vertices)

    return geometry.Polygon(
        vertices=vertices, count=safe_count, ring_counts=final_rings, overflow=overflow
    )


@jax.jit(static_argnames=["max_vertices", "max_rings"])
def difference(
    polygon1: geometry.Polygon,
    polygon2: geometry.Polygon,
    max_vertices: int = 256,
    max_rings: int = 16,
) -> geometry.Polygon:
    """Computes the difference P1 - P2.

    Args:
        polygon1: Subject polygon.
        polygon2: Clip polygon.
        max_vertices: Size of the output buffer.
        max_rings: Maximum number of output rings.

    Returns:
        The difference polygon.
    """

    def extract_edges(poly):
        N = poly.vertices.shape[0]
        c = poly.count
        rings = poly.ring_counts
        n_rings = rings.shape[0]
        starts = jnp.cumsum(jnp.pad(rings, (1, 0))[:-1])
        idxs = jnp.arange(max_vertices)

        def get_seg(i):
            idx_next = core._get_ring_aware_next_index(i, starts, rings, n_rings)
            p1 = poly.vertices[i]
            p2 = poly.vertices[idx_next]

            is_after = i >= starts
            k = jnp.sum(is_after) - 1
            k = jnp.maximum(0, jnp.minimum(k, n_rings - 1))
            rc = rings[k]
            s = starts[k]
            local = i - s

            valid = (i < c) & (local < rc) & (rc > 0)
            return jnp.stack([p1, p2]), valid

        segs, valids = jax.vmap(get_seg)(idxs)
        return segs, valids

    seg1, valid1 = extract_edges(polygon1)
    buf1 = jnp.zeros_like(seg1)

    def pack1(s, k):
        b, p = s
        b = b.at[p].set(jnp.where(valid1[k], seg1[k], b[p]))
        p = p + jnp.where(valid1[k], 1, 0)
        return (b, p), None

    (seg1, count1), _ = jax.lax.scan(pack1, (buf1, 0), jnp.arange(max_vertices))

    # 1. P1 clipped by P2 (Keep Outside)
    # Global Clip against ALL of P2
    seg1, count1 = core._clip_segments_multiring(
        seg1,
        count1,
        polygon2.vertices,
        polygon2.count,
        polygon2.ring_counts,
        max_vertices,
        keep_inside=False,
    )

    # 2. P2 clipped by P1 (Keep Inside)
    # The segments of P2 that are INSIDE P1 form the other part of the difference boundary?
    # Difference(A, B) = Boundary(A - B)
    # = Parts of A outside B + Parts of B inside A (reversed)
    # Wait, B is a HOLE in the result.
    # Yes, edges of B inside A need to be included (as hole edges).
    # And they should be reversed.

    seg2_raw, valid2 = extract_edges(polygon2)
    buf2 = jnp.zeros_like(seg2_raw)

    def pack2(s, k):
        b, p = s
        b = b.at[p].set(jnp.where(valid2[k], seg2_raw[k], b[p]))
        p = p + jnp.where(valid2[k], 1, 0)
        return (b, p), None

    (seg2, count2), _ = jax.lax.scan(pack2, (buf2, 0), jnp.arange(max_vertices))

    # Clip P2 against P1 (Keep INSIDE P1)
    seg2, count2 = core._clip_segments_multiring(
        seg2,
        count2,
        polygon1.vertices,
        polygon1.count,
        polygon1.ring_counts,
        max_vertices,
        keep_inside=True,
    )

    # Reverse P2 segments (Hole orientation)
    seg2 = seg2[:, ::-1, :]

    capacity = max_vertices * 2
    all_segments = jnp.zeros((capacity * 2, 2, 2))
    all_segments = jax.lax.dynamic_update_slice(all_segments, seg1, (0, 0, 0))
    all_segments = jax.lax.dynamic_update_slice(all_segments, seg2, (count1, 0, 0))

    all_count = count1 + count2

    vertices, count, final_rings = core._stitch_segments(
        all_segments, all_count, max_vertices, max_rings
    )

    input_ovf = (polygon1.count > max_vertices) | (polygon2.count > max_vertices)
    res_ovf = count >= max_vertices
    overflow = input_ovf | res_ovf

    vertices, count = core._clean_vertices(vertices, count)
    safe_count = jnp.minimum(count, max_vertices)

    return geometry.Polygon(
        vertices=vertices, count=safe_count, ring_counts=final_rings, overflow=overflow
    )
