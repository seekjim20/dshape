"""Core geometry helper functions."""

import jax
import jax.numpy as jnp
import geometry

# type hints
from jax import Array
from jax.typing import ArrayLike


def _clean_vertices(vertices, count):
    """Removes consecutive duplicate vertices."""
    # Mask valid vertices
    mask = jnp.arange(vertices.shape[0]) < count

    # Compare with previous vertex
    # Roll vertices to compare i with i-1
    prev_verts = jnp.roll(vertices, 1, axis=0)

    # First vertex is always kept (unless count=0, but loop logic handles it?)
    # Actually, jnp.roll wraps around. so vertex 0 compares with vertex N-1.
    # This removes closure duplicate? Yes tailored for closed loops.

    diff = jnp.abs(vertices - prev_verts).sum(axis=1)
    is_distinct = diff > 1e-6

    # Always keep vertex 0? No, if it's same as end, we might keep it if needed for closure representation?
    # But usually we want Unique vertices.
    # Wait, dshape Polygon usually implies implicit closure.
    # So if last == first, we should remove last.
    # But this logic does 'consecutive duplicates'.

    # Let's keep existing logic from ops.py implies
    # (Checking diff > 1e-6)

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


def _is_point_in_polygon(point, vertices, count):
    """Ray casting point-in-polygon test."""
    # Simple crossing number algorithm
    x, y = point

    indices = jnp.arange(vertices.shape[0])
    next_indices = jnp.where(indices + 1 == count, 0, indices + 1)

    v1 = vertices[indices]
    v2 = vertices[next_indices]

    # Check edge (v1, v2)
    # Condition: (v1.y > y) != (v2.y > y) and x < ...

    cond1 = (v1[:, 1] > y) != (v2[:, 1] > y)

    # Intersection x
    # x_int = (v2.x - v1.x) * (y - v1.y) / (v2.y - v1.y) + v1.x
    slope = (v2[:, 0] - v1[:, 0]) / (v2[:, 1] - v1[:, 1] + 1e-9)
    x_int = slope * (y - v1[:, 1]) + v1[:, 0]

    cond2 = x < x_int

    crossings = jnp.sum((cond1 & cond2) & (indices < count))

    return (crossings % 2) == 1


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


@jax.jit(static_argnames=["max_v"])
def _stitch_segments(segments: ArrayLike, count: int, max_v: int) -> tuple[Array, int]:
    """Stitches segments into a continuous polygon loop.

    This function attempts to form a closed loop from a set of line segments.
    It supports bridging distinct components (holes/islands) using a proximity heuristic.

    Args:
        segments: Array of shape (N, 2, 2) containing line segments (start, end).
        count: Number of valid segments.
        max_v: Maximum number of vertices in the output.

    Returns:
        A tuple (vertices, vertex_count).
    """
    starts = segments[:, 0, :]
    ends = segments[:, 1, :]
    seg_mask = jnp.arange(segments.shape[0]) < count

    # Start at max X (guaranteed to be on outer boundary if mostly convex-ish)
    safe_x = jnp.where(seg_mask, starts[:, 0], -1e9)
    start_idx = jnp.argmax(safe_x)

    out_verts = jnp.zeros((max_v, 2))

    # Initial segment selection
    curr_seg = segments[start_idx]
    out_verts = out_verts.at[0].set(curr_seg[0])
    curr_point = curr_seg[1]

    used_mask = jnp.zeros(segments.shape[0], dtype=bool)
    used_mask = used_mask.at[start_idx].set(True)

    out_ptr = 1

    # Stack for saving state when bridging to components
    # (used for backtracking or loop closure)
    bridge_stack = jnp.zeros((5, 2))
    stack_ptr = 0

    # Helper for 2D cross product
    def cross_2d(a, b):
        return a[0] * b[1] - a[1] * b[0]

    # Initial prev_p: extrapolate back from first segment
    initial_prev_p = curr_seg[0]

    # Loop state: (curr_p, prev_p, used_mask, buf_ptr, buf, stack, stack_ptr)
    init_state = (
        curr_point,
        initial_prev_p,
        used_mask,
        out_ptr,
        out_verts,
        bridge_stack,
        stack_ptr,
        jnp.zeros(max_v),  # area_hist: accumulated area at each index
        jnp.array(0.0),  # curr_area: current scalar area
    )

    def step(state, step_idx):
        curr_p, prev_p, mask, ptr, buf, stack, sp, area_hist, curr_area = state

        # 1. Candidate Selection
        # Find unused segments starting close to curr_p
        dists = jnp.linalg.norm(starts - curr_p, axis=1)

        # Use tight tolerance to prevent skipping small segments in smooth arcs
        valid_cand = (dists < 1e-5) & (~mask) & (jnp.arange(segments.shape[0]) < count)

        # 2. Scoring (Left-turn preference vs Velocity)
        vec_in = curr_p - prev_p
        vec_out = ends - starts

        scores = cross_2d(vec_in, vec_out.T)
        scores = jnp.where(valid_cand, scores, -1e9)

        next_idx = jnp.argmax(scores)
        found = valid_cand[next_idx]

        # 3. Strategy Selection
        # Case 1: Continue (Found valid segment)
        # Case 2: Backtrack (Pop stack)
        # Case 3: Bridge (Jump to closest unused segment)

        # Find closest unused segment for bridging
        any_unused = (~mask) & (jnp.arange(segments.shape[0]) < count)
        unused_dists = jnp.where(any_unused, dists, 1e9)
        closest_unused_idx = jnp.argmin(unused_dists)
        min_dist = unused_dists[closest_unused_idx]

        found_unused = min_dist < 1e5
        has_stack = sp > 0

        # Scenario Target Points
        s1_next_p = ends[next_idx] if segments.shape[0] > 0 else curr_p
        s1_mask_idx = next_idx

        s2_next_p = stack[sp - 1]

        s3_next_p = starts[closest_unused_idx] if segments.shape[0] > 0 else curr_p

        # Determine case priority
        case = 0
        case = jnp.where(found_unused, 3, case)
        case = jnp.where(has_stack, 2, case)
        case = jnp.where(found, 1, case)

        # Update Next Point
        next_p = curr_p
        next_p = jnp.where(case == 1, s1_next_p, next_p)
        next_p = jnp.where(case == 2, s2_next_p, next_p)
        next_p = jnp.where(case == 3, s3_next_p, next_p)

        # Update Mask (mark segment as used)
        mask_idx = -1
        mask_idx = jnp.where(case == 1, s1_mask_idx, mask_idx)
        mask = jnp.where(mask_idx != -1, mask.at[mask_idx].set(True), mask)

        # Update Buffer
        should_write = case > 0
        buf = buf.at[ptr].set(curr_p)
        ptr = ptr + jnp.where(should_write, 1, 0)

        # Update Stack
        # Case 3: Push current point as return target
        # Case 2: Pop
        stack = stack.at[sp].set(jnp.where(case == 3, curr_p, stack[sp]))
        sp = sp + jnp.where(case == 3, 1, 0)
        sp = sp - jnp.where(case == 2, 1, 0)

        next_p = jnp.where(case == 0, curr_p, next_p)
        new_prev_p = curr_p

        # --- Self-Intersection Pruning ---
        # Detect if adding edge (curr_p, next_p) creates a loop with existing path.

        # Helper to check intersection with past edges in buffer
        def check_intersection(i, _ptr, _buf, _p_start, _p_end):
            # Check edge i: (_buf[i], _buf[i+1])
            p1 = _buf[i]
            p2 = _buf[i + 1]

            # Use robust line intersection check
            # Utilizing _line_intersection helper from module scope would be ideal
            # but we inline relevant checks for performance/closure safety.

            # Orientation function
            def orientation(a, b, c):
                return (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])

            o1 = orientation(p1, p2, _p_start)
            o2 = orientation(p1, p2, _p_end)
            o3 = orientation(_p_start, _p_end, p1)
            o4 = orientation(_p_start, _p_end, p2)

            # Strict crossing check
            # Use tighter tolerance to avoid false positives on valid bridging or touching segments
            intersect = (o1 * o2 < -1e-7) & (o3 * o4 < -1e-7)

            # Compute intersection point
            pt = _line_intersection(p1, p2, _p_start, _p_end)

            # Ignore recent edges (immediate neighbors)
            valid_idx = i < _ptr - 2
            return intersect & valid_idx, pt

        # Scan previous edges
        prune_indices = jnp.arange(max_v)
        is_intersect_v, pts_v = jax.vmap(
            check_intersection, in_axes=(0, None, None, None, None)
        )(prune_indices, ptr, buf, curr_p, next_p)

        has_int = jnp.any(is_intersect_v)

        # Update Area History
        # Add trapezoidal area of segment (curr_p -> next_p)
        area_delta = 0.5 * (curr_p[0] * next_p[1] - curr_p[1] * next_p[0])
        new_area = curr_area + area_delta

        # We record the accumulated area at 'ptr' (where next_p will live)
        # Note: We need this stored BEFORE we might prune.
        new_area_hist = area_hist.at[ptr].set(new_area)

        # Resolve Intersection
        # Find earliest intersection index (smallest k) to identify the loop
        valid_int_indices = jnp.where(is_intersect_v, prune_indices, max_v + 1)
        target_k = jnp.min(valid_int_indices)

        safe_k = jnp.where(has_int, target_k, 0)
        target_pt = pts_v[safe_k]

        # Heuristic: Keep the larger component (Head vs Loop) based on AREA
        # Head: 0...target_k (closed by target_pt)
        # Loop: target_k...ptr (closed by target_pt)

        # Estimate Head Area: area_hist[target_k]
        # Estimate Loop Area: new_area - area_hist[target_k]
        # (ignoring precise closure area terms as they are roughly comparable)

        area_head_est = new_area_hist[target_k]
        area_loop_est = new_area - area_head_est

        keep_loop = jnp.abs(area_loop_est) > jnp.abs(area_head_est)

        # Only prune self-intersections for natural segment continuation (Case 1).
        # We disable pruning for Bridging (Case 3) to prevent corrupting disjoint components
        # (forcing a bad bridge is better than cutting the polygon).
        # We also disable for Backtracking (Case 2) as it follows known paths.
        do_prune = (case == 1) & has_int

        # Case A: Keep Head (Prune Loop)
        # Reset ptr to target_k + 1, set buf[target_k] = intersection
        ptr_A = target_k + 1
        buf_A = buf.at[target_k].set(target_pt)

        # Case B: Keep Loop (Prune Head)
        # Shift loop to start: buf[0] = intersection, buf[1...] = buf[target_k+1...]
        shift_amt = target_k
        buf_shifted = jnp.roll(buf, -shift_amt, axis=0)
        ptr_B = ptr - shift_amt
        buf_B = buf_shifted.at[0].set(target_pt)

        # Apply choice
        new_ptr = ptr
        new_ptr = jnp.where(do_prune & (~keep_loop), ptr_A, new_ptr)
        new_ptr = jnp.where(do_prune & keep_loop, ptr_B, new_ptr)

        buf = jnp.where(do_prune & keep_loop, buf_B, buf)
        buf = jnp.where(do_prune & (~keep_loop), buf_A, buf)

        ptr = new_ptr

        # Update history point for next iteration
        # If we pruned, we effectively jumped to 'target_pt'.
        # For 'Keep Loop', target_pt closes the loop.
        # For 'Keep Head', we continue to 'next_p' or stay at 'target_pt'?
        # We usually continue tracing from the intersection.

        # Original logic:
        # new_prev_p_A = next_p
        # new_prev_p_B = target_pt

        new_prev_p = jnp.where(do_prune & keep_loop, target_pt, new_prev_p)
        new_prev_p = jnp.where(do_prune & (~keep_loop), next_p, new_prev_p)

        # Fix Stack if pruned (cancel incomplete moves)
        sp = jnp.where(do_prune & (case == 3), sp - 1, sp)
        sp = jnp.where(do_prune & (case == 2), sp + 1, sp)

        return (
            next_p,
            new_prev_p,
            mask,
            ptr,
            buf,
            stack,
            sp,
            new_area_hist,
            new_area,
        ), None

    final_state, _ = jax.lax.scan(step, init_state, jnp.arange(max_v - 1))

    _, _, _, final_ptr, final_buf, _, _, _, _ = final_state

    return final_buf, final_ptr


def _intersection(
    polygon1: ArrayLike, polygon2: ArrayLike, max_vertices: int
) -> tuple[Array, int]:
    """Computes the intersection of two polygons using the Sutherland-Hodgman algorithm.

    Assumptions:
        - polygon2 is convex and vertices are in counter-clockwise order.
        - polygon1 can be concave but results might need triangulation for some uses (though SH usually outputs a valid polygon for convex clipper).
        - max_vertices is sufficient to hold the result.

    Args:
        polygon1: Subject polygon (N, 2).
        polygon2: Clip polygon (M, 2) - MUST BE CONVEX and CCW.
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

    # If we clamped the input, we are already overflowing logic-wise if we cared about strictness.
    # But checking input overflow is separate. Here we just prevent crash.

    # State for the scan over clip edges:
    # (current_subject_vertices, current_count)
    init_state = (padded_subject, curr_len)

    def clip_edge_scan_body(state, clip_edge):
        # clip_edge is ((2,), (2,)) representing (cp1, cp2)
        in_vertices, in_count = state
        cp1, clip_p2 = clip_edge

        # We need to create the next set of vertices
        out_vertices = jnp.zeros((max_vertices, 2))
        out_count = 0

        # Iterate over input vertices
        # We need pairs (prev, curr).
        # Since in_vertices is padded, we need to handle the wrapping carefully based on in_count.

        # To make this JIT-friendly, we scan over the *maximum* possible edges of in_vertices.
        # But we only care up to in_count.

        def vertex_step(inner_state, i):
            out_buf, w_idx = inner_state

            # Indices for prev and curr
            # prev index is (i - 1) % in_count
            # curr index is i
            # But modulo with dynamic in_count is tricky if we want strict static bounds,
            # but here we iterate `i` from 0 to max_vertices - 1.
            # We only act if i < in_count.

            curr_idx = i
            prev_idx = jnp.where(i == 0, in_count - 1, i - 1)

            curr_v = in_vertices[curr_idx]
            prev_v = in_vertices[prev_idx]

            # Logic conditions
            # 1. Check if valid processing (i < in_count)
            is_valid_step = i < in_count

            curr_in = _is_inside(curr_v, cp1, clip_p2)
            prev_in = _is_inside(prev_v, cp1, clip_p2)

            # Intersection point
            # For gradients to flow, we compute intersection even if not needed strictly, or mask it?
            # Actually we only use it if needed.
            intersect_p = _line_intersection(prev_v, curr_v, cp1, clip_p2)

            # Cases:
            # 1. Both inside: add curr
            # 2. First inside, second outside: add intersection
            # 3. Second inside, first outside: add intersection, add curr
            # 4. Both outside: do nothing

            # We need to append 0, 1, or 2 vertices.

            # Case 1: prev_in & curr_in -> Add curr
            # Case 2: prev_in & !curr_in -> Add intersection
            # Case 3: !prev_in & curr_in -> Add Intersection, Add Curr
            # Case 4: !prev_in & !curr_in -> Do nothing

            transition = prev_in != curr_in

            # If transition, we add intersection.
            # If curr_in, we add curr.

            # Case 3 (!prev_in, curr_in): Add Intersection, then Add Curr.
            # First potential addition: Intersection (if transition)
            # Second potential addition: Curr (if curr_in)

            # We use `w_idx` to write.

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

        return (new_vertices, new_count), None

    # Prepare clip edges for scan
    # Shift clip polygon to get pairs
    clip_p1 = clip_polygon
    indices = jnp.arange(clip_polygon.shape[0])
    next_indices = jnp.where(indices + 1 >= clip_polygon.shape[0], 0, indices + 1)
    clip_p2 = clip_polygon[next_indices]

    # We zip them
    clip_edges = (clip_p1, clip_p2)

    # Scan over clip edges
    scan_result, _ = jax.lax.scan(clip_edge_scan_body, init_state, clip_edges)

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
    # Capacity for intermediate segments.
    # Max segments likely <= max_out_verts * 2.
    # We define capacity for the expanded segments.
    # OLD: capacity = 200  # Fixed internal capacity for safety
    # NEW: Use static dynamic size based on output requirement
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


def _clip_segments(
    segments: ArrayLike,
    count: int,
    clip_verts: ArrayLike,
    clip_count: int,
    max_out_segments: int,
    keep_inside: bool = True,
) -> tuple[Array, int]:
    """Clips a set of segments against a polygon.

    Args:
        segments: Input segments (N, 2, 2).
        count: Number of valid segments.
        clip_verts: Clip polygon vertices (M, 2).
        clip_count: Number of valid clip vertices.
        max_out_segments: Maximum number of output segments.
        keep_inside: If True, keep parts inside the clip polygon. Else outside.

    Returns:
        tuple: (output_segments, output_count)
    """
    # 1. Expand segments by intersecting with clip edges
    # We maintain segments in (K, 2, 2) format.

    # To reuse _insert_intersections, we need to adapt it.
    # _insert_intersections takes (p1, c1, p2, c2, max_v) and outputs vertices.
    # It assumes p1 is a connected loop.
    # Our 'segments' are disconnected.

    # We can write a specialized segment-clipper.

    # A segment (A, B) intersected by clip polygon edges might become multiple segments (A, I1), (I1, I2), (I2, B).
    # Then we filter them based on midpoint.

    # Step 1: Find intersections of EACH segment with ALL clip edges.
    # Collect all points (start, end, intersections) on the segment line.
    # Sort them by distance from start.
    # Form sub-segments.

    # Implementation details:
    # Iterate segments. For each segment:
    #   Find intersections with clip_verts.
    #   Sort intersections.
    #   Create sub-segments: (A, I1), (I1, I2)... (Im, B).
    #   Check midpoint of each sub-segment.
    #   Write valid ones to output buffer.

    out_buf = jnp.zeros((max_out_segments, 2, 2))
    out_ptr = 0

    def process_segment(state, i):
        buf, ptr = state

        # Current segment
        seg = segments[i]
        p_start = seg[0]
        p_end = seg[1]

        is_valid_seg = i < count

        # Vector
        v_seg = p_end - p_start
        len_seg = jnp.linalg.norm(v_seg)

        # Find intersections with clip edges
        def get_intersection(j):
            c_idx1 = j
            c_idx2 = jnp.where(j + 1 == clip_count, 0, j + 1)
            cp1 = clip_verts[c_idx1]
            cp2 = clip_verts[c_idx2]

            p_int = _line_intersection(p_start, p_end, cp1, cp2)

            # Check on both segments
            def on_seg_strict(p, a, b):
                d = jnp.linalg.norm(a - b)
                d1 = jnp.linalg.norm(a - p)
                d2 = jnp.linalg.norm(p - b)
                return jnp.abs(d1 + d2 - d) < 1e-6

            valid = on_seg_strict(p_int, p_start, p_end) & on_seg_strict(
                p_int, cp1, cp2
            )
            valid = valid & (j < clip_count)

            dist = jnp.linalg.norm(p_int - p_start)
            return p_int, valid, dist

        # Scan clip edges (limit 100)
        scan_limit = 100
        ints, valids, dists = jax.vmap(get_intersection)(jnp.arange(scan_limit))

        # Add start (dist 0) and end (dist len_seg) to the list of points
        # to form simple intervals.

        # We need to sort points: Start, Int1, Int2... End.
        # Let's verify we have capacity.
        # Max intersections?

        # Pack candidates: Start, End, Inte...
        # Candidates: (MAX_INT + 2)
        MAX_INT = 10

        cand_points = jnp.zeros((MAX_INT + 2, 2))
        cand_dists = jnp.zeros((MAX_INT + 2))
        cand_valids = jnp.zeros((MAX_INT + 2), dtype=bool)

        # Set Start
        cand_points = cand_points.at[0].set(p_start)
        cand_dists = cand_dists.at[0].set(0.0)
        cand_valids = cand_valids.at[0].set(True)

        # Set End
        cand_points = cand_points.at[1].set(p_end)
        cand_dists = cand_dists.at[1].set(len_seg)
        cand_valids = cand_valids.at[1].set(True)

        # Fill intersections
        # Sort indices of intersections by distance
        dists_masked = jnp.where(valids, dists, 1e9)
        perm = jnp.argsort(dists_masked)

        def fill_int(k):
            idx = perm[k]
            return ints[idx], valids[idx], dists[idx]

        # Take top MAX_INT intersections
        v_fill = jax.vmap(fill_int)(jnp.arange(MAX_INT))

        cand_points = cand_points.at[2:].set(v_fill[0])
        cand_valids = cand_valids.at[2:].set(v_fill[1])
        cand_dists = cand_dists.at[2:].set(v_fill[2])

        # Now sort ALL candidates by distance
        # Mask invalids to huge distance
        sort_dists = jnp.where(cand_valids, cand_dists, 1e9)
        final_perm = jnp.argsort(sort_dists)

        sorted_points = cand_points[final_perm]
        sorted_valids = cand_valids[final_perm]

        # Create sub-segments
        # (p[k], p[k+1])
        # Valid if valid[k] and valid[k+1] and distance < huge

        # Capacity of sub-segments: MAX_INT + 1

        def check_subseg(k):
            # sub-segment from k to k+1
            sp1 = sorted_points[k]
            sp2 = sorted_points[k + 1]

            is_real = sorted_valids[k] & sorted_valids[k + 1] & (k < MAX_INT + 1)

            # Additional check: sort_dists[k+1] should be < 1e8
            is_real = is_real & (sort_dists[k + 1] < 1e8)

            # Check length > tiny
            slen = jnp.linalg.norm(sp2 - sp1)
            is_real = is_real & (slen > 1e-6)

            mid = (sp1 + sp2) * 0.5
            vec = sp2 - sp1
            length = jnp.linalg.norm(vec)
            length = jnp.where(length < 1e-9, 1.0, length)
            # Outward normal (y, -x)
            normal = jnp.array([vec[1], -vec[0]]) / length

            test_p = mid + normal * 1e-5

            # Containment check
            # For robustness, use probe
            # But line is exactly on boundary? No, crossing only at endpoints.
            # Midpoint is strictly inside or outside usually.

            # Using 1e-6 check to match recent fixes
            is_in = _is_point_in_polygon(test_p, clip_verts, clip_count)
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


def _offset_vertex(
    p: ArrayLike,
    n1: ArrayLike,
    n2: ArrayLike,
    is_convex: bool,
    dist: ArrayLike,
    max_pts: int,
) -> tuple[Array, int]:
    """Generates offset vertices for a corner P with incoming normal n1 and outgoing n2.

    Args:
        p: Vertex position (2,).
        n1: Incoming edge normal (2,).
        n2: Outgoing edge normal (2,).
        is_convex: Boolean indicating if corner is convex.
        dist: Buffer distance.
        max_pts: Maximum points to generate (e.g., for arc).

    Returns:
        A tuple (points, count). points has shape (max_pts, 2).
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
    # Used when: (Convex & Erosion) OR (Concave & Dilation) OR Parallel
    use_intersection = (
        (is_convex & (dist < 0)) | ((~is_convex) & (dist > 0)) | is_parallel
    )

    def branch_miter(_):
        # Result: 1 point
        pts = jnp.zeros((max_pts, 2))
        pts = pts.at[0].set(miter_pt)
        return pts, 1

    def branch_arc(_):
        # Generate arc points
        n_arc = 8
        # Ensure n_arc <= max_pts.

        fracs = jnp.arange(n_arc, dtype=jnp.float32) / (n_arc - 1)

        # Determine angle direction
        d_final = diff
        d_final = jnp.where(is_convex & (diff < 0), diff + 2 * jnp.pi, d_final)
        d_final = jnp.where((~is_convex) & (diff > 0), diff - 2 * jnp.pi, d_final)

        thetas = ang1 + fracs * d_final

        c = jnp.cos(thetas)
        s = jnp.sin(thetas)

        # Points: p + r * (cos, sin)
        r = dist

        arc_pts = p + r * jnp.stack([c, s], axis=1)

        # Pad to max_pts
        out_a = jnp.zeros((max_pts, 2))
        out_a = out_a.at[:n_arc].set(arc_pts)

        return out_a, n_arc

    return jax.lax.cond(use_intersection, branch_miter, branch_arc, None)


def _buffer(
    vertices: ArrayLike, count: int, distance: ArrayLike, max_vertices: int
) -> tuple[Array, int]:
    # 1. Compute Normals
    n = vertices.shape[0]
    indices = jnp.arange(n)

    prev_indices = jnp.where(indices == 0, count - 1, indices - 1)
    next_indices = jnp.where(indices + 1 == count, 0, indices + 1)

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

    verts_per_corner = 16

    def process_corner(p, n_prev, n_next, convex, d):
        return _offset_vertex(p, n_prev, n_next, convex, d, verts_per_corner)

    # generated_chunks: (MaxIn, 16, 2)
    # chunk_counts: (MaxIn,)
    corner_poly_fn = jax.vmap(process_corner, in_axes=(0, 0, 0, 0, None))
    generated_chunks, chunk_counts = corner_poly_fn(p, n1, n2, is_convex, distance)

    # Mask invalid corners (i >= count)
    valid_corner_mask = indices < count
    chunk_counts = jnp.where(valid_corner_mask, chunk_counts, 0)

    # 4. Pack chunks into temporary buffer (optional, mostly for debug/visualization if needed)
    out_buf = jnp.zeros((max_vertices, 2))

    def write_chunk(state, pkg):
        buf, ptr = state
        chunk, n_pts = pkg

        def write_pt(p_state, k):
            b, p = p_state
            val = chunk[k]
            do_write = k < n_pts
            b = b.at[p].set(jnp.where(do_write, val, b[p]))
            p = p + jnp.where(do_write, 1, 0)
            return (b, p), None

        (buf, ptr), _ = jax.lax.scan(write_pt, (buf, ptr), jnp.arange(verts_per_corner))
        return (buf, ptr), None

    (final_buf, final_count), _ = jax.lax.scan(
        write_chunk, (out_buf, 0), (generated_chunks, chunk_counts)
    )

    # 5. Extract Segments for Stitching
    # A. Intra-chunk segments
    capacity = max_vertices * 2

    def extract_intra_segments(chunk, cnt, i):
        def get_seg(j):
            p1 = chunk[j]
            p2 = chunk[j + 1]
            valid = (j < cnt - 1) & (i < count)
            return jnp.stack([p1, p2]), valid

        segs, valids = jax.vmap(get_seg)(jnp.arange(verts_per_corner - 1))
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
        idx_next = jnp.where(i + 1 == count, 0, i + 1)

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
            & (~is_inverted_edge[i])
            & (cnt_i > 0)
            & (chunk_counts[idx_next] > 0)
        )

        return seg, valid

    inter_segs, inter_valids = jax.vmap(extract_inter_segment)(indices)

    # Combine all segments
    all_segs_list = [intra_segs, inter_segs]
    all_valids_list = [intra_valids, inter_valids]

    total_segs = jnp.concatenate(all_segs_list, axis=0)
    total_valids = jnp.concatenate(all_valids_list, axis=0)

    # Pack segments for stitcher
    out_seg_buf = jnp.zeros((capacity, 2, 2))

    def compress_step(state, x):
        buf, ptr = state
        seg, valid = x
        buf = buf.at[ptr].set(jnp.where(valid, seg, buf[ptr]))
        ptr = ptr + jnp.where(valid, 1, 0)
        return (buf, ptr), None

    (final_seg_buf, final_seg_count), _ = jax.lax.scan(
        compress_step, (out_seg_buf, 0), (total_segs, total_valids)
    )

    # 6. Stitch
    vertices, count = _stitch_segments(final_seg_buf, final_seg_count, max_vertices)

    return vertices, count
