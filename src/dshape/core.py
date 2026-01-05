"""Core geometry helper functions."""

import jax
import jax.numpy as jnp
from . import geometry

# type hints
from jax import Array
from jax.typing import ArrayLike


def _clean_vertices(vertices, count):
    """Removes consecutive duplicate vertices."""
    # Mask valid vertices
    mask = jnp.arange(vertices.shape[0]) < count

    # Compare with previous vertex to remove consecutive duplicates
    prev_verts = jnp.roll(vertices, 1, axis=0)
    diff = jnp.abs(vertices - prev_verts).sum(axis=1)
    is_distinct = diff > 1e-6

    keep = is_distinct & mask

    # Compact
    # Scan logic to pack
    out_verts = jnp.zeros_like(vertices)

    def pack(state, i):
        buf, ptr = state
        v = vertices[i]
        valid = keep[i]

        buf = buf.at[ptr].set(jnp.where(valid, v, buf[ptr]))
        ptr = ptr + jnp.where(valid, 1, 0)
        return (buf, ptr), None

    (new_verts, new_count), _ = jax.lax.scan(
        pack, (out_verts, 0), jnp.arange(vertices.shape[0])
    )

    return new_verts, new_count


def _get_ring_aware_next_index(i, ring_starts, ring_counts, num_rings):
    """Returns the index of the next vertex in the same ring."""
    # Find which ring 'i' belongs to
    is_after = i >= ring_starts
    k = jnp.sum(is_after) - 1
    k = jnp.maximum(0, jnp.minimum(k, num_rings - 1))

    s = ring_starts[k]
    c = ring_counts[k]
    local = i - s
    safe_c = jnp.maximum(c, 1)
    next_local = (local + 1) % safe_c
    return s + next_local


def _is_point_in_polygon_multiring(point, vertices, count, ring_counts):
    """Ray casting point-in-polygon test supporting multiple rings (Even-Odd)."""
    x, y = point

    if ring_counts is None:
        ring_counts = jnp.array([count])

    num_rings = ring_counts.shape[0]
    starts = jnp.cumsum(jnp.pad(ring_counts, (1, 0))[:-1])

    indices = jnp.arange(vertices.shape[0])

    # Calculate next indices respecting rings
    next_indices = jax.vmap(
        lambda i: _get_ring_aware_next_index(i, starts, ring_counts, num_rings)
    )(indices)

    v1 = vertices[indices]
    v2 = vertices[next_indices]

    cond1 = (v1[:, 1] > y) != (v2[:, 1] > y)
    slope = (v2[:, 0] - v1[:, 0]) / (v2[:, 1] - v1[:, 1] + 1e-9)
    x_int = slope * (y - v1[:, 1]) + v1[:, 0]
    cond2 = x < x_int

    # Check validity:
    # i < count checks if we are in valid vertex range.
    # ALSO: Implicitly, if i is in a valid ring, then next_indices[i] is in same ring.
    # But if ring_counts specifies fewer vertices than 'count', we might process garbage?
    # Usually count == sum(ring_counts).
    # But let's stick to i < count.

    is_valid_edge = indices < count

    crossings = jnp.sum((cond1 & cond2) & is_valid_edge)

    return (crossings % 2) == 1


# Wraps old compatible signature for single ring usage if needed?
# Or we just update usages.
# Existing _is_point_in_polygon signature: (point, vertices, count)
# We can make ring_counts optional.
def _is_point_in_polygon(point, vertices, count, ring_counts=None):
    return _is_point_in_polygon_multiring(point, vertices, count, ring_counts)


def _line_intersection(p1, p2, p3, p4):
    """Computes intersection of line segments p1-p2 and p3-p4."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)

    # If parallel, denom is 0. Avoid division by zero.
    denom = jnp.where(jnp.abs(denom) < 1e-9, 1e-9, denom)

    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom

    x = x1 + ua * (x2 - x1)
    y = y1 + ua * (y2 - y1)

    return jnp.append(jnp.atleast_1d(x), jnp.atleast_1d(y))


def _is_inside(p, cp1, cp2):
    """Checks if point p is inside the half-plane defined by edge cp1->cp2 (Left side)."""
    return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) - (cp2[1] - cp1[1]) * (
        p[0] - cp1[0]
    ) >= 0


def _remove_duplicate_segments(segments: ArrayLike, count: int) -> tuple[Array, int]:
    """Removes duplicate segments from the list.

    Args:
        segments: (N, 2, 2) array of segments.
        count: current valid count.

    Returns:
        (filtered_segments, new_count)
    """
    N = segments.shape[0]

    # We want to mask out duplicates.
    # Keep the first occurrence.
    # Mask[i] = True if unique.

    # O(N^2) comparison.
    indices = jnp.arange(N)

    def check_is_duplicate(i):
        # Check if i is a duplicate of any k < i
        seg_i = segments[i]

        def check_k(k):
            seg_k = segments[k]
            # dist between starts
            d1 = jnp.linalg.norm(seg_i[0] - seg_k[0])
            # dist between ends
            d2 = jnp.linalg.norm(seg_i[1] - seg_k[1])

            is_same = (d1 < 1e-5) & (d2 < 1e-5)
            # Check if valid
            is_active = (k < i) & (k < count)
            return is_same & is_active

        matches = jax.vmap(check_k)(indices)
        has_duplicate = jnp.any(matches)

        return has_duplicate

    is_dup = jax.vmap(check_is_duplicate)(indices)

    keep = (~is_dup) & (indices < count)

    # Pack
    out_buf = jnp.zeros_like(segments)

    def pack(state, i):
        buf, ptr = state
        should_keep = keep[i]
        buf = buf.at[ptr].set(jnp.where(should_keep, segments[i], buf[ptr]))
        ptr = ptr + jnp.where(should_keep, 1, 0)
        return (buf, ptr), None

    (final_buf, final_count), _ = jax.lax.scan(pack, (out_buf, 0), indices)

    return final_buf, final_count


@jax.jit(static_argnames=["max_out"])
def _extract_edges(
    vertices: ArrayLike, count: int, ring_counts: ArrayLike, max_out: int
) -> tuple[Array, int]:
    """Extracts segments from vertices."""
    # (N, 2, 2)
    # Iterate rings.

    # Precompute starts
    # Pad rcs for safety (though it's usually fixed size)
    rcs = ring_counts
    starts = jnp.cumsum(jnp.pad(rcs, (1, 0))[:-1])
    num_rings = rcs.shape[0]

    # We want to generate indices for each segment.
    # Total segments = sum(ring_counts).
    # Vertices buffer is N.
    # We can just iterate all vertices i < count.
    # For each i, find next vertex j.
    # If i is last in ring, j is start of ring.

    # Find which ring i belongs to.
    # Binary search or just scan starts? num_rings is small (16).

    def get_seg_indices(i):
        # Only valid if i < count

        # Find ring k
        # starts[k] <= i < starts[k] + rc[k]
        is_after = i >= starts  # (16,)
        # Last True is the ring index.
        k = jnp.sum(is_after) - 1
        # Clamp k to valid range just in case
        k = jnp.maximum(0, jnp.minimum(k, num_rings - 1))

        s = starts[k]
        rc = rcs[k]

        # Local index
        local = i - s

        # Check if i is actually in valid range of this ring
        # If ring is empty/invalid, rc=0.
        # But i < count ensures we are in *some* valid data range, assuming packed?
        # Yes, vertices are packed.

        # Next index
        safe_rc = jnp.maximum(rc, 1)
        next_local = (local + 1) % safe_rc
        next_abs = s + next_local

        return i, next_abs

    # vmap
    indices = jnp.arange(max_out)  # Buffer size
    curr_idx, next_idx = jax.vmap(get_seg_indices)(indices)

    # Fetch verts
    curr_v = vertices[curr_idx]  # (max_out, 2)
    next_v = vertices[next_idx]

    # Stack segments
    segments = jnp.stack([curr_v, next_v], axis=1)  # (max_out, 2, 2)

    # Count is same as vertex count (since polygons are closed loops)
    # Just ensure we cap at max_out
    safe_count = jnp.minimum(count, max_out)

    return segments, safe_count


@jax.jit(static_argnames=["max_v", "max_rings"])
def _stitch_segments(
    segments: ArrayLike,
    count: int,
    max_v: int,
    max_rings: int = 16,
    seg_inverted_mask: ArrayLike = None,
) -> tuple[Array, int, Array]:
    """Stitches segments into multiple continuous polygon rings.

    Args:
        segments: Array of shape (N, 2, 2) containing line segments (start, end).
        count: Number of valid segments.
        max_v: Maximum number of vertices in the output.
        max_rings: Maximum number of rings to detect.
        seg_inverted_mask: Optional boolean mask (N,) indicating inverted segments.
                           Rings with >50% inverted segments are discarded (Ghost Rings).

    Returns:
        A tuple (vertices, vertex_count, ring_counts).
    """
    seg_starts = segments[:, 0, :]

    if seg_inverted_mask is None:
        seg_inverted_mask = jnp.zeros(segments.shape[0], dtype=bool)

    # Mask for used segments
    # 0 = Unused, 1 = Used
    used_mask = jnp.zeros(segments.shape[0], dtype=bool)

    out_verts = jnp.zeros((max_v, 2))
    ring_counts = jnp.zeros(max_rings, dtype=jnp.int32)

    # State: (mode, curr_p, start_p, used_mask, ptr, buf, ring_idx, ring_cnt, ring_counts, ring_start_ptr, r_inv_cnt)

    def find_start_seg(mask):
        valid_idxs = (~mask) & (jnp.arange(segments.shape[0]) < count)
        has_any = jnp.any(valid_idxs)

        # Pick max X among valid to normalize start (helps consistency)
        xs = seg_starts[:, 0]
        safe_xs = jnp.where(valid_idxs, xs, -1e9)
        best_idx = jnp.argmax(safe_xs)
        return best_idx, has_any

    init_idx, has_first = find_start_seg(used_mask)
    init_mode = jnp.where(has_first, 1, 0)

    # Init Mode 1 Trace vars
    curr_seg = segments[init_idx]
    p_start = curr_seg[0]
    p_curr = curr_seg[1]

    first_inv = seg_inverted_mask[init_idx]

    used_mask = used_mask.at[init_idx].set(True)
    out_verts = out_verts.at[0].set(p_start)

    # Ptr 0 written. Next is 1.
    # Ring started at 0.
    buf_ptr = 1
    ring_cnt = 1
    ring_idx = 0
    ring_start_ptr = 0
    r_inv_cnt = jnp.where(first_inv, 1, 0)

    init_state = (
        init_mode,
        p_curr,
        p_start,
        used_mask,
        buf_ptr,
        out_verts,
        ring_idx,
        ring_cnt,
        ring_counts,
        ring_start_ptr,
        r_inv_cnt,
    )

    def step(state, _):
        (
            mode,
            curr_p,
            start_p,
            mask,
            ptr,
            buf,
            r_idx,
            r_cnt,
            r_counts,
            r_start_ptr,
            inv_c,
        ) = state

        # --- Mode 0: Search ---
        new_start_idx, found_new = find_start_seg(mask)

        m0_seg = segments[new_start_idx]
        m0_p_start = m0_seg[0]
        m0_p_curr = m0_seg[1]
        m0_inv = seg_inverted_mask[new_start_idx]

        # Valid transition check
        can_start = (mode == 0) & found_new & (ptr < max_v) & (r_idx < max_rings)

        # Transition updates
        m0_start_ptr = ptr
        m0_buf = buf.at[ptr].set(m0_p_start)
        m0_ptr = ptr + 1
        m0_mask = mask.at[new_start_idx].set(True)
        m0_r_cnt = 1
        m0_inv_c = jnp.where(m0_inv, 1, 0)

        # --- Mode 1: Trace ---
        dist_to_start = jnp.linalg.norm(curr_p - start_p)
        is_closed = dist_to_start < 1e-4

        dists = jnp.linalg.norm(seg_starts - curr_p, axis=1)
        cand_mask = (dists < 1e-4) & (~mask) & (jnp.arange(segments.shape[0]) < count)
        has_cand = jnp.any(cand_mask)
        cand_idx = jnp.argmax(cand_mask)

        finish_ring = (mode == 1) & (is_closed | (~has_cand))

        # Check Ghost (Partial Inversion Ratio)
        # Avoid div by zero
        safe_rc = jnp.maximum(r_cnt, 1)
        inv_ratio = inv_c / safe_rc
        is_ghost = inv_ratio > 0.5

        # Action: Finish Ring
        # If ghost, discard (reset ptr). Else keep.
        drop_ring = finish_ring & is_ghost
        keep_ring = finish_ring & (~is_ghost)

        fr_ptr = jnp.where(drop_ring, r_start_ptr, ptr)
        fr_r_idx = jnp.where(drop_ring, r_idx, r_idx + 1)
        fr_r_counts = jnp.where(drop_ring, r_counts, r_counts.at[r_idx].set(r_cnt))

        # Action: Continue Trace
        ct_seg = segments[cand_idx]
        ct_p_curr = ct_seg[1]
        ct_inv = seg_inverted_mask[cand_idx]

        ct_mask = mask.at[cand_idx].set(True)
        ct_buf = buf.at[ptr].set(curr_p)
        ct_ptr = ptr + 1
        ct_r_cnt = r_cnt + 1
        ct_inv_c = inv_c + jnp.where(ct_inv, 1, 0)

        # Select Next State
        next_mode = mode
        # 0 -> 1
        next_mode = jnp.where((mode == 0) & can_start, 1, next_mode)
        # 1 -> 0
        next_mode = jnp.where((mode == 1) & finish_ring, 0, next_mode)

        # Updates
        # 0 -> 1
        condition_0_1 = (mode == 0) & can_start
        curr_p = jnp.where(condition_0_1, m0_p_curr, curr_p)
        start_p = jnp.where(condition_0_1, m0_p_start, start_p)
        mask = jnp.where(condition_0_1, m0_mask, mask)
        ptr = jnp.where(condition_0_1, m0_ptr, ptr)
        buf = jnp.where(condition_0_1, m0_buf, buf)
        r_cnt = jnp.where(condition_0_1, m0_r_cnt, r_cnt)
        r_start_ptr = jnp.where(condition_0_1, m0_start_ptr, r_start_ptr)
        inv_c = jnp.where(condition_0_1, m0_inv_c, inv_c)

        # 1 -> 0 (Back to Search)
        condition_1_0 = (mode == 1) & finish_ring
        r_counts = jnp.where(condition_1_0, fr_r_counts, r_counts)
        r_idx = jnp.where(condition_1_0, fr_r_idx, r_idx)
        ptr = jnp.where(condition_1_0, fr_ptr, ptr)  # Reset ptr if ghost

        # 1 -> 1 (Continue)
        condition_1_1 = (mode == 1) & (~finish_ring)
        curr_p = jnp.where(condition_1_1, ct_p_curr, curr_p)
        mask = jnp.where(condition_1_1, ct_mask, mask)
        buf = jnp.where(condition_1_1, ct_buf, buf)
        ptr = jnp.where(condition_1_1, ct_ptr, ptr)
        r_cnt = jnp.where(condition_1_1, ct_r_cnt, r_cnt)
        inv_c = jnp.where(condition_1_1, ct_inv_c, inv_c)

        return (
            next_mode,
            curr_p,
            start_p,
            mask,
            ptr,
            buf,
            r_idx,
            r_cnt,
            r_counts,
            r_start_ptr,
            inv_c,
        ), None

    final_state, _ = jax.lax.scan(step, init_state, jnp.arange(max_v))

    (
        _,
        _,
        _,
        _,
        final_ptr,
        final_buf,
        final_r_idx,
        final_r_cnt,
        final_r_counts,
        _,
        _,
    ) = final_state

    return final_buf, final_ptr, final_r_counts


def _intersection(
    polygon1: ArrayLike,
    count1: int,
    polygon2: ArrayLike,
    count2: int,
    max_vertices: int,
) -> tuple[Array, int]:
    """Computes the intersection of two polygons using the Sutherland-Hodgman algorithm.

    Arguments:
        polygon1: Subject polygon (N, 2).
        count1: Number of valid vertices in polygon1.
        polygon2: Clip polygon (M, 2) - MUST BE CONVEX and CCW.
        count2: Number of valid vertices in polygon2.
        max_vertices: Maximum number of vertices for the result buffer.

    Returns:
        A tuple containing:
            clipped_polygon: The intersected polygon vertices (max_vertices, 2).
            valid_count: The number of valid vertices.
    """

    subject_polygon = polygon1
    clip_polygon = polygon2

    # We maintain a buffer of vertices. Initial subject polygon.
    # Pad to max_vertices
    # CLAMPING: We must limit the input to max_vertices to prevent OOB.
    input_len = subject_polygon.shape[0]
    # Use python min to keep it static for slicing
    curr_len = min(input_len, max_vertices)

    padded_subject = jnp.zeros((max_vertices, 2))
    padded_subject = padded_subject.at[:curr_len].set(subject_polygon[:curr_len])

    # We use count1 to limit processing of subject.
    # scan state: (current_subject_vertices, current_count)
    # Clamp count1 to curr_len
    safe_count1 = jnp.minimum(count1, curr_len)

    init_state = (padded_subject, safe_count1)

    # We scan over CLIP edges (up to count2)
    # We need to supply clip edges.

    # Since we need to iterate exactly `count2` edges, but scan requires static length,
    # we iterate `max_clip` (shape[0]) and mask invalid steps.

    max_clip = clip_polygon.shape[0]

    # Helper for clip edge
    def get_clip_edge(j):
        c2 = count2
        safe_c2 = jnp.maximum(c2, 1)  # avoid mod 0
        curr_idx = j
        next_idx = (j + 1) % safe_c2

        p1 = clip_polygon[curr_idx]
        p2 = clip_polygon[next_idx]

        valid = j < c2
        return (p1, p2), valid

    def clip_edge_scan_body(state, j):
        in_vertices, in_count = state

        # Get clip edge
        (cp1, clip_p2), is_valid_clip = get_clip_edge(j)

        # We need to create the next set of vertices
        out_vertices = jnp.zeros((max_vertices, 2))
        out_count = 0

        def vertex_step(inner_state, i):
            out_buf, w_idx = inner_state

            # Wrapping
            # in_count is dynamic.
            # prev index = (i - 1 + in_count) % in_count
            safe_cnt = jnp.maximum(in_count, 1)

            curr_idx = i
            prev_idx = (i - 1 + safe_cnt) % safe_cnt

            # Since in_vertices is padded, we must ensure we read valid data?
            # We trust in_count.

            curr_v = in_vertices[curr_idx]
            prev_v = in_vertices[prev_idx]

            # Logic conditions
            # 1. Check if valid processing (i < in_count)
            is_valid_step = i < in_count

            # Optimization: If clip edge is invalid (padding), we should NOT reduce `in_vertices`?
            # Sutherland-Hodgman operates sequentially.
            # If we skip a clip step, we pass input to output unmodified.
            # So if `~is_valid_clip`, we just copy `curr_v`?
            # Let's handle is_valid_clip at top level of body?
            # Yes.

            curr_in = _is_inside(curr_v, cp1, clip_p2)
            prev_in = _is_inside(prev_v, cp1, clip_p2)

            # Intersection
            intersect_p = _line_intersection(prev_v, curr_v, cp1, clip_p2)

            transition = prev_in != curr_in

            # Case: prev_in & curr_in -> Add curr
            # Case (!prev_in, curr_in): Add Intersection, Add Curr.
            # Case (prev_in, !curr_in): Add Intersection

            # Write intersection
            should_write_int = is_valid_step & transition
            out_buf = out_buf.at[w_idx].set(
                jnp.where(should_write_int, intersect_p, out_buf[w_idx])
            )
            w_idx = w_idx + jnp.where(should_write_int, 1, 0)

            # Write curr
            should_write_curr = is_valid_step & curr_in
            out_buf = out_buf.at[w_idx].set(
                jnp.where(should_write_curr, curr_v, out_buf[w_idx])
            )
            w_idx = w_idx + jnp.where(should_write_curr, 1, 0)

            return (out_buf, w_idx), None

        # Scan over vertices
        (new_vertices, new_count), _ = jax.lax.scan(
            vertex_step, (out_vertices, out_count), jnp.arange(max_vertices)
        )

        # If this clip edge was invalid, we keep the OLD vertices (input state)
        # effectively identity op.
        final_vertices = jnp.where(is_valid_clip, new_vertices, in_vertices)
        final_count = jnp.where(is_valid_clip, new_count, in_count)

        return (final_vertices, final_count), None

    # Scan over clip edges
    scan_result, _ = jax.lax.scan(clip_edge_scan_body, init_state, jnp.arange(max_clip))

    final_vertices, final_count = scan_result
    return final_vertices, final_count


def _insert_intersections(p1, c1, p2, c2, max_v):
    # Insert intersection points of p1 edges with p2 edges into p1.
    out_verts = jnp.zeros((max_v, 2))
    out_ptr = 0

    def edge_scan(state, i):
        ptr, buf = state

        s_idx1 = i
        s_idx2 = jnp.where(i + 1 == c1, 0, i + 1)  # Scan up to c1, but safer to modulo

        sp1 = p1[s_idx1]
        sp2 = p1[s_idx2]

        # Find intersections with ALL clip edges
        def clip_interaction(j):
            c_idx1 = j
            c_idx2 = jnp.where(j + 1 == c2, 0, j + 1)
            cp1 = p2[c_idx1]
            cp2 = p2[c_idx2]

            p_int = _line_intersection(sp1, sp2, cp1, cp2)

            # Check if on segment
            def on_seg(p, a, b):
                d_ab = jnp.linalg.norm(a - b)
                d_ap = jnp.linalg.norm(a - p)
                d_pb = jnp.linalg.norm(p - b)
                return jnp.abs(d_ap + d_pb - d_ab) < 1e-6

            valid = on_seg(p_int, sp1, sp2) & on_seg(p_int, cp1, cp2)
            valid = valid & (j < c2)

            dist = jnp.linalg.norm(p_int - sp1)
            return p_int, valid, dist

        clip_max = 100  # hardcoded max scan for loop unroll or static scan
        # We assume p2 fits in 100.

        ints, valids, dists = jax.vmap(clip_interaction)(jnp.arange(clip_max))

        # Sort these intersections by distance
        # For simplicity, if we have multiple, we just add them.
        # But order matters for line following.
        # Let's simple check if ANY valid
        # Actually proper set ops require precise ordering.
        # This function is a bit approximate if multiple intersections occur on one segment.
        # For now, let's take the closest ONE if any?
        # Or sorting.
        # Sorting in JAX is fine.

        # Mask invalid dists to infinity
        dists = jnp.where(valids, dists, 1e9)
        perm = jnp.argsort(dists)

        # How many intersections?
        num_ints = jnp.sum(valids)

        # Write start point
        should_write_start = i < c1
        buf = buf.at[ptr].set(jnp.where(should_write_start, sp1, buf[ptr]))
        ptr = ptr + jnp.where(should_write_start, 1, 0)

        # Write intersections in order
        def write_int(pk_state, k):
            b, p = pk_state
            idx = perm[k]
            # Must be valid intersection AND valid edge
            is_valid_int = (dists[idx] < 1e8) & (i < c1)

            b = b.at[p].set(jnp.where(is_valid_int, ints[idx], b[p]))
            p = p + jnp.where(is_valid_int, 1, 0)
            return (b, p), None

        # Max intersections per edge e.g. 5
        (buf, ptr), _ = jax.lax.scan(write_int, (buf, ptr), jnp.arange(5))

        return (ptr, buf), None

    # Scan bound: iterate over input edges.
    # We use max_v as input capacity bounds.
    (final_ptr, final_buf), _ = jax.lax.scan(
        edge_scan, (0, out_verts), jnp.arange(max_v)
    )

    return final_buf, final_ptr


def _get_clipped_segments(
    subject_verts,
    subject_count,
    clip_verts,
    clip_count,
    max_out_verts,
    keep_inside=True,
):
    capacity = max_out_verts * 2

    # 1. Expand Subject Edge to Sub-segments
    exp_verts, exp_count = _insert_intersections(
        subject_verts, subject_count, clip_verts, clip_count, capacity
    )

    # 2. Filter Segments
    def map_seg(i):
        idx1 = i
        idx2 = (i + 1) % exp_count
        p1 = exp_verts[idx1]
        p2 = exp_verts[idx2]

        mid = (p1 + p2) * 0.5

        # Calculate outward normal
        vec = p2 - p1
        length = jnp.linalg.norm(vec)
        # Normal for CCW polygon is (v.y, -v.x)?
        # Edge vector (x, y). Rot -90 is (y, -x) -> Right Turn?
        # CCW polygon... tangent is along edge.
        # "Left" is inside. "Right" is outside.
        # So Outward Normal is (y, -x).

        # To be safe, let's normalize
        normal = jnp.array([vec[1], -vec[0]]) / (length + 1e-9)

        # Test point slightly outside
        test_p = mid + normal * 1e-6

        # Check if inside
        is_in = _is_point_in_polygon(test_p, clip_verts, clip_count)

        # Union: keep_inside=False (Keep segments OUTSIDE the clip polygon)
        # Difference: keep_inside=True (Keep segments INSIDE the clip polygon)

        keep = is_in == keep_inside
        valid = (i < exp_count) & keep

        seg = jnp.stack([p1, p2])
        return seg, valid

    indices = jnp.arange(capacity)
    segments, valids = jax.vmap(map_seg)(indices)

    # Compress/Pack
    out_buf = jnp.zeros((capacity, 2, 2))

    def compress_step(state, x):
        buf, ptr = state
        seg, valid = x

        buf = buf.at[ptr].set(jnp.where(valid, seg, buf[ptr]))
        ptr = ptr + jnp.where(valid, 1, 0)
        return (buf, ptr), None

    (final_buf, final_count), _ = jax.lax.scan(
        compress_step, (out_buf, 0), (segments, valids)
    )

    return final_buf, final_count


def _clip_segments_multiring(
    segments: ArrayLike,
    count: int,
    clip_verts: ArrayLike,
    clip_count: int,
    clip_ring_counts: ArrayLike,
    max_out_segments: int,
    keep_inside: bool = True,
) -> tuple[Array, int]:
    """Clips a set of segments against a multi-ring polygon.

    Args:
        segments: Input segments (N, 2, 2).
        count: Number of valid segments.
        clip_verts: Clip polygon vertices (M, 2).
        clip_count: Number of valid clip vertices.
        clip_ring_counts: Ring counts for clip polygon.
        max_out_segments: Maximum number of output segments.
        keep_inside: If True, keep parts inside the clip polygon. Else outside.

    Returns:
        tuple: (output_segments, output_count)
    """
    if clip_ring_counts is None:
        clip_ring_counts = jnp.array([clip_count])

    clip_num_rings = clip_ring_counts.shape[0]
    clip_starts = jnp.cumsum(jnp.pad(clip_ring_counts, (1, 0))[:-1])

    # Precompute clip indices map
    # We iterate over all clip verts to find edges
    clip_indices = jnp.arange(clip_verts.shape[0])

    # Use helper to find next index for edge construction
    clip_next_indices = jax.vmap(
        lambda i: _get_ring_aware_next_index(
            i, clip_starts, clip_ring_counts, clip_num_rings
        )
    )(clip_indices)

    # OUTPUT BUFFER
    out_buf = jnp.zeros((max_out_segments, 2, 2))
    out_ptr = 0

    def process_segment(state, i):
        buf, ptr = state

        # Current segment
        seg = segments[i]
        p_start = seg[0]
        p_end = seg[1]

        # Valid segment?
        is_valid_seg = i < count

        # Vector
        v_seg = p_end - p_start
        len_seg = jnp.linalg.norm(v_seg)

        # 1. INTERSECTION FINDING
        # We check against ALL edges of the clip polygon
        def get_intersection(j):
            c_idx1 = j
            c_idx2 = clip_next_indices[j]  # Use ring-aware next

            cp1 = clip_verts[c_idx1]
            cp2 = clip_verts[c_idx2]

            p_int = _line_intersection(p_start, p_end, cp1, cp2)

            # Strict containment on input segment
            def on_seg_strict(p, a, b):
                d = jnp.linalg.norm(a - b)
                d1 = jnp.linalg.norm(a - p)
                d2 = jnp.linalg.norm(p - b)
                return jnp.abs(d1 + d2 - d) < 1e-6

            # Valid if: Use both intersection checks + valid edge index j
            valid = on_seg_strict(p_int, p_start, p_end) & on_seg_strict(
                p_int, cp1, cp2
            )
            # Edge j is valid if j < clip_count
            valid = valid & (j < clip_count)

            dist = jnp.linalg.norm(p_int - p_start)
            return p_int, valid, dist

        # Scan limit: Assume clip polygon fits in some bound or use clip_verts size.
        # We can scan over all vertices in clip_verts buffer.
        # Assuming max clip verts is e.g. 256 or derived from shape.
        scan_limit = clip_verts.shape[0]

        ints, valids, dists = jax.vmap(get_intersection)(jnp.arange(scan_limit))

        # 2. SORT INTERSECTIONS
        MAX_INT = 16  # Increased capacity for complex clips

        cand_points = jnp.zeros((MAX_INT + 2, 2))
        cand_dists = jnp.zeros((MAX_INT + 2))
        cand_valids = jnp.zeros((MAX_INT + 2), dtype=bool)

        # Start/End
        cand_points = cand_points.at[0].set(p_start)
        cand_dists = cand_dists.at[0].set(0.0)
        cand_valids = cand_valids.at[0].set(True)

        cand_points = cand_points.at[1].set(p_end)
        cand_dists = cand_dists.at[1].set(len_seg)
        cand_valids = cand_valids.at[1].set(True)

        # Top K intersections
        dists_masked = jnp.where(valids, dists, 1e9)
        perm = jnp.argsort(dists_masked)

        def fill_int(k):
            idx = perm[k]
            return ints[idx], valids[idx], dists[idx]

        v_fill = jax.vmap(fill_int)(jnp.arange(MAX_INT))

        cand_points = cand_points.at[2:].set(v_fill[0])
        cand_valids = cand_valids.at[2:].set(v_fill[1])
        cand_dists = cand_dists.at[2:].set(v_fill[2])

        # Sort all candidates by distance along segment
        sort_dists = jnp.where(cand_valids, cand_dists, 1e9)
        final_perm = jnp.argsort(sort_dists)

        sorted_points = cand_points[final_perm]
        sorted_valids = cand_valids[final_perm]

        # 3. GENERATE SUB-SEGMENTS & CHECK CONTAINMENT
        def check_subseg(k):
            # Segment k -> k+1
            sp1 = sorted_points[k]
            sp2 = sorted_points[k + 1]

            # Valid if both points are valid and distance sensible
            # And within count
            is_real = sorted_valids[k] & sorted_valids[k + 1] & (k < MAX_INT + 1)
            is_real = is_real & (sort_dists[k + 1] < 1e8)

            slen = jnp.linalg.norm(sp2 - sp1)
            is_real = is_real & (slen > 1e-6)

            # Midpoint Check
            mid = (sp1 + sp2) * 0.5

            # Use normal offset for robustness?
            # Midpoint is safer than offset if we trust Even-Odd.
            # If line is ON edge, Even-Odd might be flaky?
            # Let's nudge slightly.
            vec = sp2 - sp1
            nvec = jnp.array([vec[1], -vec[0]])
            # Normalized
            nvec = nvec / (slen + 1e-9)

            # Test point
            test_p = mid + nvec * 1e-5

            # MULTI-RING CONTAINMENT CHECK
            is_in = _is_point_in_polygon_multiring(
                test_p, clip_verts, clip_count, clip_ring_counts
            )

            should_keep = is_in == keep_inside

            final_valid = is_real & should_keep & is_valid_seg

            return jnp.stack([sp1, sp2]), final_valid

        sub_segs, sub_valids = jax.vmap(check_subseg)(jnp.arange(MAX_INT + 1))

        # Write to buffer
        def write_sub(w_state, k):
            b, p = w_state
            ss = sub_segs[k]
            sv = sub_valids[k]
            b = b.at[p].set(jnp.where(sv, ss, b[p]))
            p = p + jnp.where(sv, 1, 0)
            return (b, p), None

        (buf, ptr), _ = jax.lax.scan(write_sub, (buf, ptr), jnp.arange(MAX_INT + 1))

        return (buf, ptr), None

    (final_buf, final_ptr), _ = jax.lax.scan(
        process_segment, (out_buf, out_ptr), jnp.arange(segments.shape[0])
    )

    return final_buf, final_ptr


# Keep old symbol for compatibility if needed, but we will likely replace usage.
# Or redirect.
def _clip_segments(
    segments: ArrayLike,
    count: int,
    clip_verts: ArrayLike,
    clip_count: int,
    max_out_segments: int,
    keep_inside: bool = True,
) -> tuple[Array, int]:
    # Redirect to multiring with default ring count
    return _clip_segments_multiring(
        segments, count, clip_verts, clip_count, None, max_out_segments, keep_inside
    )


def _check_edge_inversion_mask(input_verts, count, chunks, chunk_counts):
    """Checks if offset edges are inverted relative to input edges.

    This detects if an edge 'vanished' or reversed direction due to erosion.

    Args:
        input_verts: Input polygon vertices (N, 2).
        count: Number of valid vertices.
        chunks: Generated corner chunks for each vertex (N, K, 2).
        chunk_counts: Number of valid points per chunk (N,).

    Returns:
        Boolean mask of shape (N,) where True indicates the edge starting at i is inverted.
    """
    # Check if offset edges run opposite to input edges.
    # Returns (MaxIn,) boolean array. True = Inverted (Invalid).

    max_v = input_verts.shape[0]
    indices = jnp.arange(max_v)

    def check_edge(i):
        # i -> i_next
        idx_next = jnp.where(i + 1 == count, 0, i + 1)

        # Input Vector
        # We need input_verts[i] and input_verts[idx_next]
        # vmap maps i. input_verts is captured.
        # This capture seems safe usually.
        # But to be consistent with intra-fix, we could pass it?
        # But input_verts is simple array.

        p_in_start = input_verts[i]
        p_in_end = input_verts[idx_next]
        v_in = p_in_end - p_in_start

        # Output Vector
        # Standard closure capture for chunks/chunk_counts might be the issue?

        c_i = chunks[i]
        cnt_i = chunk_counts[i]
        # Safe Indexing
        last_idx = jnp.maximum(0, cnt_i - 1)
        p_start = c_i[last_idx]

        # Neighbor chunks[idx_next]
        c_next = chunks[idx_next]
        p_end = c_next[0]

        v_out = p_end - p_start

        # Dot product
        dot = jnp.dot(v_in, v_out)

        is_opp = dot < -1e-6

        return is_opp

    # We must be careful about captured variables.
    return jax.vmap(check_edge)(indices)


@jax.jit(static_argnames=["resolution"])
def _offset_vertex(
    p: ArrayLike,
    n1: ArrayLike,
    n2: ArrayLike,
    is_convex: bool,
    dist: ArrayLike,
    resolution: int,
) -> tuple[Array, int]:
    """Generates offset vertices for a corner P with incoming normal n1 and outgoing n2.

    Args:
        p: Vertex position (2,).
        n1: Incoming edge normal (2,).
        n2: Outgoing edge normal (2,).
        is_convex: Boolean indicating if corner is convex.
        dist: Buffer distance.
        resolution: Resolution of the buffer operation (segments per circle).
                    Also acts as the max buffer size for this corner.

    Returns:
        A tuple (points, count). points has shape (resolution, 2).
    """

    # Shifted lines:
    # L1: p + n1 * dist + t * (tangent of n1)
    # L2: p + n2 * dist + t * (tangent of n2)

    # Point on L1 offset: p1 = p + n1 * dist
    # Point on L2 offset: p2 = p + n2 * dist
    # Intersection logic helper
    t1 = jnp.array([n1[1], -n1[0]])
    t2 = jnp.array([n2[1], -n2[0]])

    # Calculate Miter Intersection
    p1_shift = p + n1 * dist
    p2_shift = p + n2 * dist
    miter_pt = _line_intersection(p1_shift, p1_shift + t1, p2_shift, p2_shift + t2)

    # Arc Generation parameters
    ang1 = jnp.arctan2(n1[1], n1[0])
    ang2 = jnp.arctan2(n2[1], n2[0])

    diff = ang2 - ang1
    # Wrap to [-pi, pi]
    diff = jnp.mod(diff + jnp.pi, 2 * jnp.pi) - jnp.pi

    # Check orientation relation with distance
    # Dilation: dist > 0.
    # Erosion: dist < 0.

    # Special case: Parallel normals (Collinear edge).
    dot = jnp.dot(n1, n2)
    is_parallel = dot > 1 - 1e-6

    # Decide Strategy
    # Case A: Miter/Intersection (Sharp corner)
    # Used when: (Convex & Erosion) OR (Concave & Dilation) OR (Concave & Erosion) OR Parallel
    # Basically everything EXCEPT (Convex & Dilation).

    use_intersection = (~(is_convex & (dist > 0))) | is_parallel

    def branch_miter(_):
        # Result: 1 point
        pts = jnp.zeros((resolution, 2))
        pts = pts.at[0].set(miter_pt)
        return pts, 1

    def branch_arc(_):
        # Determine angle direction and magnitude
        d_final = diff
        d_final = jnp.where(is_convex & (diff < 0), diff + 2 * jnp.pi, d_final)
        d_final = jnp.where((~is_convex) & (diff > 0), diff - 2 * jnp.pi, d_final)

        # Adaptive number of segments
        # Resolution is segments per full circle (2pi)
        # n = ceil( abs(angle) / (2pi) * resolution )
        n_float = jnp.ceil(jnp.abs(d_final) / (2 * jnp.pi) * resolution)
        n_arc = n_float.astype(jnp.int32)

        # Clamp to [1, resolution] (should naturally be <= resolution since abs(d_final) <= 2pi)
        n_arc = jnp.maximum(1, jnp.minimum(n_arc, resolution))

        # Ensure n_arc <= resolution.

        fracs = jnp.arange(resolution, dtype=jnp.float32) / (jnp.maximum(n_arc - 1, 1))
        # Mask valid fracs
        # We only need first n_arc points

        thetas = ang1 + fracs * d_final

        c = jnp.cos(thetas)
        s = jnp.sin(thetas)

        # Points: p + r * (cos, sin)
        r = dist

        arc_pts = p + r * jnp.stack([c, s], axis=1)

        # Zero out invalid points (masking done implicitly by return count)
        # But for cleanliness, let's keep array as is. Valid range is 0..n_arc-1

        return arc_pts, n_arc

    return jax.lax.cond(use_intersection, branch_miter, branch_arc, None)


@jax.jit(static_argnames=["max_vertices", "resolution"])
def _buffer(
    vertices: ArrayLike,
    count: int,
    ring_counts: ArrayLike,
    distance: ArrayLike,
    max_vertices: int,
    resolution: int = 60,
) -> tuple[Array, int, Array]:

    # Handle optional ring_counts
    if ring_counts is None:
        ring_counts = jnp.array([count])

    num_rings = ring_counts.shape[0]
    starts = jnp.cumsum(jnp.pad(ring_counts, (1, 0))[:-1])

    # 1. Compute Normals
    n_verts = vertices.shape[0]
    indices = jnp.arange(n_verts)

    # We need prev and next indices for normal calculation, per ring.
    def get_indices(i):
        # find ring
        is_after_start = i >= starts
        k = jnp.sum(is_after_start) - 1
        k = jnp.maximum(0, jnp.minimum(k, num_rings - 1))

        s = starts[k]
        c = ring_counts[k]
        local = i - s
        safe_c = jnp.maximum(c, 1)

        # Next
        next_local = (local + 1) % safe_c
        next_abs = s + next_local

        # Prev
        prev_local = (local - 1 + safe_c) % safe_c
        prev_abs = s + prev_local

        return prev_abs, next_abs

    prev_indices, next_indices = jax.vmap(get_indices)(indices)

    p = vertices
    p_prev = vertices[prev_indices]
    p_next = vertices[next_indices]

    # Vectors
    v1 = p - p_prev
    v2 = p_next - p

    def get_normal(vec):
        l = jnp.linalg.norm(vec)
        # Prevent div by zero
        l = jnp.where(l < 1e-9, 1.0, l)
        return jnp.array([vec[1], -vec[0]]) / l

    n1 = jax.vmap(get_normal)(v1)
    n2 = jax.vmap(get_normal)(v2)

    # 2. Determine Convexity
    # Cross product of v1 and v2
    # v1 x v2 = v1.x * v2.y - v1.y * v2.x
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]

    # Convex if Left Turn (Cross > 0)
    is_convex = cross >= -1e-9

    # 3. Generate Corner Segments
    # For each vertex, generate miter or arc

    # We use resolution as the max size for the corner chunk
    # verts_per_corner = 16  <-- Removed, use resolution

    def process_corner(p, n_prev, n_next, convex, d):
        return _offset_vertex(p, n_prev, n_next, convex, d, resolution)

    # generated_chunks: (MaxIn, resolution, 2)
    # chunk_counts: (MaxIn,)
    corner_poly_fn = jax.vmap(process_corner, in_axes=(0, 0, 0, 0, None))
    generated_chunks, chunk_counts = corner_poly_fn(p, n1, n2, is_convex, distance)

    # Mask invalid corners (i >= count)
    valid_corner_mask = indices < count
    chunk_counts = jnp.where(valid_corner_mask, chunk_counts, 0)

    # 5. Extract Segments for Stitching
    # A. Intra-chunk segments
    capacity = max_vertices * 2

    def extract_intra_segments(chunk, cnt, i):
        def get_seg(j):
            p1 = chunk[j]
            p2 = chunk[j + 1]
            valid = (j < cnt - 1) & (i < count)
            return jnp.stack([p1, p2]), valid

        segs, valids = jax.vmap(get_seg)(jnp.arange(resolution - 1))
        return segs, valids

    intra_segs, intra_valids = jax.vmap(extract_intra_segments)(
        generated_chunks, chunk_counts, indices
    )

    intra_segs = intra_segs.reshape(-1, 2, 2)
    intra_valids = intra_valids.reshape(-1)

    # B. Inter-chunk segments (connect end of corner i to start of corner i+1)
    is_inverted_edge = _check_edge_inversion_mask(
        p, count, generated_chunks, chunk_counts
    )

    def extract_inter_segment(i):
        # i -> i_next
        idx_next = next_indices[i]  # Use precomputed ring-aware next index

        # Last point of i
        c_i = generated_chunks[i]
        cnt_i = chunk_counts[i]
        last_idx = jnp.maximum(0, cnt_i - 1)
        p_start = c_i[last_idx]

        # First point of next
        c_next = generated_chunks[idx_next]
        p_end = c_next[0]

        seg = jnp.stack([p_start, p_end])

        valid = (
            (i < count)
            # We don't verify rings match because next_indices[i] GUARANTEES same ring
            # & (~is_inverted_edge[i])  <-- DISABLED FILTER to prevent gaps in topology
            & (cnt_i > 0)
            & (chunk_counts[idx_next] > 0)
        )

        return seg, valid

    inter_segs, inter_valids = jax.vmap(extract_inter_segment)(indices)

    # Inversion metadata
    # Intra segments (corners) are never inverted
    intra_invs = jnp.zeros(intra_segs.shape[0], dtype=bool)
    inter_invs = is_inverted_edge  # (N,)

    # Combine all segments
    all_segs_list = [intra_segs, inter_segs]
    all_valids_list = [intra_valids, inter_valids]
    all_invs_list = [intra_invs, inter_invs]

    total_segs = jnp.concatenate(all_segs_list, axis=0)
    total_valids = jnp.concatenate(all_valids_list, axis=0)
    total_invs = jnp.concatenate(all_invs_list, axis=0)

    # Pack segments for stitcher
    out_seg_buf = jnp.zeros((capacity, 2, 2))
    out_inv_buf = jnp.zeros(capacity, dtype=bool)

    def compress_step(state, x):
        buf, inv_buf, ptr = state
        seg, valid, inv = x

        buf = buf.at[ptr].set(jnp.where(valid, seg, buf[ptr]))
        inv_buf = inv_buf.at[ptr].set(jnp.where(valid, inv, inv_buf[ptr]))

        ptr = ptr + jnp.where(valid, 1, 0)
        return (buf, inv_buf, ptr), None

    (final_seg_buf, final_inv_buf, final_seg_count), _ = jax.lax.scan(
        compress_step,
        (out_seg_buf, out_inv_buf, 0),
        (total_segs, total_valids, total_invs),
    )

    # 6. Stitch
    vertices, count, out_ring_counts = _stitch_segments(
        final_seg_buf, final_seg_count, max_vertices, seg_inverted_mask=final_inv_buf
    )

    # Overflow if generated segments exceeded max_vertices
    # OR if stitch output was capped
    stop_overflow = count >= max_vertices
    seg_overflow = final_seg_count > max_vertices
    overflow = stop_overflow | seg_overflow

    return vertices, count, out_ring_counts, overflow


def _generate_offset_chunks(vertices, indices, count, max_vertices, distance):
    """Helper to generate offset chunks (corners) for buffering.

    Refactored from _buffer to allow reuse in robust erosion.
    """
    # 1. Identify Neighbors
    # Assume default single ring context if called directly?
    # Actually _buffer passed us indices and ring topology is implicit in how neighbors are found.
    # But here we need to RE-IMPLEMENT neighbor finding?
    # Or expect the caller to pass neighbors?
    # To keep the signature simple and allow _buffer to just call it, we should verify what inputs we have.
    # _buffer uses 'vertices', 'ring_counts'.
    # Here we only recieved `vertices`, `indices`, `count`. WE MISS `ring_counts`!

    # We should probably pass ring_counts or just inline the neighbor logic inside _buffer and extract the REST.
    # Actually, extracting logic which depends on ring_counts without passing efficient structures is annoying.
    # But `_buffer` implementation had neighbor finding inside.

    # Let's revert extracting neighbor finding, and just extract the "Geometry Generation" part
    # (Normals + Corners).

    # Wait, neighbor finding is needed for Normals.

    # Let's abort the extraction via `replace_file_content` if we missed dependencies.
    # I will construct the function to be self-contained but it needs ring_counts.
    pass


def _convex_hull(
    vertices: ArrayLike, count: int, max_vertices: int
) -> tuple[Array, int]:
    """Computes the convex hull of a set of points using the Monotone Chain algorithm.

    Args:
        vertices: Input points (N, 2).
        count: Number of valid points.
        max_vertices: Buffer size for output.

    Returns:
        (hull_vertices, hull_count)
    """
    # 1. Filter valid vertices
    indices = jnp.arange(vertices.shape[0])
    # Set invalid vertices to Infinity so they sort to end
    masked_verts = jnp.where(
        indices[:, None] < count, vertices, jnp.array([jnp.inf, jnp.inf])
    )

    # 2. Sort points
    # Lexicographical sort (primary X, secondary Y)
    # jnp.lexsort((Y, X)) -> sorts by X (last key) then Y?
    # No, lexsort(keys) -> Sort by keys[-1] (primary).
    # So we want primary X. Keys should be (Y, X).
    x = masked_verts[:, 0]
    y = masked_verts[:, 1]

    sort_idx = jnp.lexsort((y, x))
    sorted_points = masked_verts[sort_idx]

    n_points = count

    # 3. Build Hull
    # Cross product (O, A, B) -> (A-O) x (B-O)
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Monotone Chain scan
    def build_chain(points_indices):
        stack = jnp.zeros((max_vertices, 2))
        ptr = 0  # Stack size

        def step(state, i):
            stk, p = state

            # Current point to consider
            curr_pt = sorted_points[i]

            # Emulate while loop with fixed-length scan for autodiff
            def pop_body(inner_state, _):
                s, pt_idx = inner_state

                # Check condition
                size_ok = pt_idx >= 2
                # Clamp indices for safety
                idx_top = jnp.maximum(0, pt_idx - 1)
                idx_prev = jnp.maximum(0, pt_idx - 2)

                p_top = s[idx_top]
                p_prev = s[idx_prev]

                cc = cross(p_prev, p_top, curr_pt)
                # <= 0 means clockwise or collinear -> Remove top
                geom_bad = cc <= 1e-7

                should_pop = size_ok & geom_bad

                new_idx = pt_idx - 1
                final_idx = jnp.where(should_pop, new_idx, pt_idx)

                return (s, final_idx), None

            # Unroll loop up to max_vertices times
            (stk, p), _ = jax.lax.scan(pop_body, (stk, p), None, length=max_vertices)

            # Push current if valid
            valid_i = i < n_points
            stk = stk.at[p].set(jnp.where(valid_i, curr_pt, stk[p]))
            p = p + jnp.where(valid_i, 1, 0)

            return (stk, p), None

        (final_stack, final_ptr), _ = jax.lax.scan(step, (stack, ptr), points_indices)
        return final_stack, final_ptr

    # Lower Hull
    indices = jnp.arange(vertices.shape[0])
    lower_stack, lower_count = build_chain(indices)

    # Upper Hull: iterate backwards
    rev_indices = jnp.flip(indices)
    upper_stack, upper_count = build_chain(rev_indices)

    # 4. Concatenate
    # L[:-1] + U[:-1]
    safe_lower_c = jnp.maximum(0, lower_count - 1)
    safe_upper_c = jnp.maximum(0, upper_count - 1)

    idx = jnp.arange(max_vertices)

    def merge_get(i):
        is_lower = i < safe_lower_c
        val_l = lower_stack[i]

        u_idx = i - safe_lower_c
        u_idx = jnp.maximum(0, jnp.minimum(u_idx, max_vertices - 1))
        val_u = upper_stack[u_idx]

        return jnp.where(is_lower, val_l, val_u)

    final_verts = jax.vmap(merge_get)(idx)

    total_count = safe_lower_c + safe_upper_c

    # Mask out OOB
    mask_valid = idx < total_count
    final_verts = jnp.where(mask_valid[:, None], final_verts, 0.0)

    final_c = jnp.minimum(total_count, max_vertices)

    return final_verts, final_c
