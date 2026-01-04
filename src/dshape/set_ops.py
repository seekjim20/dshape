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
            is_after = i >= starts
            k = jnp.sum(is_after) - 1
            k = jnp.maximum(0, jnp.minimum(k, n_rings - 1))

            s = starts[k]
            rc = rings[k]
            local = i - s
            safe_rc = jnp.maximum(rc, 1)
            next_local = (local + 1) % safe_rc
            next_abs = s + next_local

            p1 = poly.vertices[i]
            p2 = poly.vertices[next_abs]

            # Valid if i < c AND rc > 0 AND local < rc
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

        # Pad for clipping scan which can produce up to max_vertices * 2 segments
        # We do this ONCE before clipping against all other polygons.
        scan_capacity = max_vertices * 2
        padded_segments = jnp.zeros((scan_capacity, 2, 2))
        padded_segments = padded_segments.at[:max_vertices].set(
            current_segments_compact
        )
        current_segments = padded_segments

        # Iterate over other polygons to clip against
        for j in range(num_polys):
            if i == j:
                continue

            clipper = polygons[j]
            c_rings = clipper.ring_counts
            c_starts = jnp.cumsum(jnp.pad(c_rings, (1, 0))[:-1])
            n_c_rings = c_rings.shape[0]

            def clip_against_ring(state, r_idx):
                segs, cnt = state
                start = c_starts[r_idx]
                rc = c_rings[r_idx]

                safe_rc = jnp.minimum(rc, max_vertices)
                read_idxs = start + jnp.arange(max_vertices)
                read_idxs = jnp.minimum(read_idxs, max_vertices - 1)

                ring_verts = clipper.vertices[read_idxs]

                run_clip = rc > 0
                eff_cnt = jnp.where(run_clip, safe_rc, 0)

                new_segs, new_cnt = core._clip_segments(
                    segs, cnt, ring_verts, eff_cnt, max_vertices * 2, keep_inside=False
                )

                return (new_segs, new_cnt), None

            (current_segments, current_count), _ = jax.lax.scan(
                clip_against_ring,
                (current_segments, current_count),
                jnp.arange(n_c_rings),
            )

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
            is_after = i >= starts
            k = jnp.sum(is_after) - 1
            k = jnp.maximum(0, jnp.minimum(k, n_rings - 1))
            s = starts[k]
            rc = rings[k]
            local = i - s
            safe_rc = jnp.maximum(rc, 1)
            next_local = (local + 1) % safe_rc
            next_abs = s + next_local
            p1 = poly.vertices[i]
            p2 = poly.vertices[next_abs]
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

    # Clip seg1 against ALL rings of P2 (Keep Outside)
    p2_rings = polygon2.ring_counts
    p2_starts = jnp.cumsum(jnp.pad(p2_rings, (1, 0))[:-1])
    n_p2_rings = p2_rings.shape[0]

    def clip_p1_step(state, r_idx):
        segs, cnt = state
        start = p2_starts[r_idx]
        rc = p2_rings[r_idx]
        safe_rc = jnp.minimum(rc, max_vertices)
        read_idxs = start + jnp.arange(max_vertices)
        read_idxs = jnp.minimum(read_idxs, max_vertices - 1)
        ring_verts = polygon2.vertices[read_idxs]
        run_clip = rc > 0
        eff_cnt = jnp.where(run_clip, safe_rc, 0)

        new_segs, new_cnt = core._clip_segments(
            segs, cnt, ring_verts, eff_cnt, max_vertices, keep_inside=False
        )
        return (new_segs, new_cnt), None

    (seg1, count1), _ = jax.lax.scan(
        clip_p1_step, (seg1, count1), jnp.arange(n_p2_rings)
    )

    # 2. P2 clipped by P1 (Keep Inside)
    seg2_raw, valid2 = extract_edges(polygon2)
    buf2 = jnp.zeros_like(seg2_raw)

    def pack2(s, k):
        b, p = s
        b = b.at[p].set(jnp.where(valid2[k], seg2_raw[k], b[p]))
        p = p + jnp.where(valid2[k], 1, 0)
        return (b, p), None

    (seg2, count2), _ = jax.lax.scan(pack2, (buf2, 0), jnp.arange(max_vertices))

    p1_rings = polygon1.ring_counts
    p1_starts = jnp.cumsum(jnp.pad(p1_rings, (1, 0))[:-1])
    n_p1_rings = p1_rings.shape[0]

    def clip_p2_step(state, r_idx):
        segs, cnt = state
        start = p1_starts[r_idx]
        rc = p1_rings[r_idx]
        safe_rc = jnp.minimum(rc, max_vertices)
        read_idxs = start + jnp.arange(max_vertices)
        read_idxs = jnp.minimum(read_idxs, max_vertices - 1)
        ring_verts = polygon1.vertices[read_idxs]

        # Calculate signed area to decide orientation
        # Use simple masking to avoid dynamic slicing
        idxs = jnp.arange(max_vertices)
        valid = idxs < safe_rc

        x = ring_verts[:, 0]
        y = ring_verts[:, 1]

        # Wrap index based on count
        next_idxs = (idxs + 1) % jnp.maximum(safe_rc, 1)

        x_next = x[next_idxs]
        y_next = y[next_idxs]

        area = 0.5 * jnp.sum((x * y_next - x_next * y) * valid)
        is_pos = area > 0
        target_keep = is_pos

        run_clip = rc > 0
        eff_cnt = jnp.where(run_clip, safe_rc, 0)

        new_segs, new_cnt = core._clip_segments(
            segs, cnt, ring_verts, eff_cnt, max_vertices, keep_inside=target_keep
        )
        return (new_segs, new_cnt), None

    (seg2, count2), _ = jax.lax.scan(
        clip_p2_step, (seg2, count2), jnp.arange(n_p1_rings)
    )

    # Reverse P2 segments
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
