"""Differentiable geometry operations.

This module provides differentiable geometric operations such as intersection,
union, and difference for polygons. It is designed to work with the Polygon
class defined in the geometry module.
"""

import jax
import jax.numpy as jnp
from geometry import Polygon


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

    overflow = input_ovf | res_ovf
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

    vertices, count = _stitch_segments(all_segments, all_count, max_vertices)

    input_ovf = (polygon1.count > max_vertices) | (polygon2.count > max_vertices)
    res_ovf = count >= max_vertices
    overflow = input_ovf | res_ovf

    safe_count = jnp.minimum(count, max_vertices)

    return Polygon(vertices=vertices, count=safe_count, overflow=overflow)


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


def _intersection(
    polygon1: jnp.ndarray, polygon2: jnp.ndarray, max_vertices: int
) -> tuple[jnp.ndarray, int]:
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
            # Case 3: !prev_in & curr_in -> Add intersection, Add Curr
            # Case 4: !prev_in & !curr_in -> Do nothing

            # To vectorize/simplify:
            # We can define candidates to add.
            # Cand 1: Intersection (if transition)
            # Cand 2: Curr (if curr_in)

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
        test_p = mid + normal * 1e-3

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


@jax.jit(static_argnames=["max_v"])
def _stitch_segments(
    segments: jnp.ndarray, count: int, max_v: int
) -> tuple[jnp.ndarray, int]:
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
    )

    def step(state, step_idx):
        curr_p, prev_p, mask, ptr, buf, stack, sp = state

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
            intersect = (o1 * o2 < -1e-5) & (o3 * o4 < -1e-5)

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

        # Resolve Intersection
        # Find earliest intersection index (smallest k) to identify the loop
        valid_int_indices = jnp.where(is_intersect_v, prune_indices, max_v + 1)
        target_k = jnp.min(valid_int_indices)

        safe_k = jnp.where(has_int, target_k, 0)
        target_pt = pts_v[safe_k]

        # Heuristic: Keep the larger component (Head vs Loop)
        # Head: 0...target_k
        # Loop: target_k...ptr
        loop_len = ptr - target_k
        head_len = target_k
        keep_loop = loop_len > head_len

        do_prune = (case > 0) & has_int

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

        return (next_p, new_prev_p, mask, ptr, buf, stack, sp), None

    final_state, _ = jax.lax.scan(step, init_state, jnp.arange(max_v - 1))

    _, _, _, final_ptr, final_buf, _, _ = final_state

    return final_buf, final_ptr


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
                return jnp.abs(d_ap + d_pb - d_ab) < 1e-4

            valid = on_seg(p_int, sp1, sp2) & on_seg(p_int, cp1, cp2)
            valid = valid & (j < c2)

            dist = jnp.linalg.norm(p_int - sp1)
            return p_int, valid, dist

        clip_max = 100  # hardcoded max scan for loop unroll or static scan
        # We assume p2 fits in 100.

        ints, valids, dists = jax.vmap(clip_interaction)(jnp.arange(clip_max))

        # Sort these intersections by distance
        perm = jnp.argsort(dists)
        sorted_ints = ints[perm]
        sorted_valids = valids[perm]

        # Append start point sp1
        buf = buf.at[ptr].set(sp1)
        ptr = ptr + 1

        # Append valid intersections
        def append_int(st, k):
            p, buf = st
            p_val = sorted_ints[k]
            is_v = sorted_valids[k]

            # Filter if too close to sp1/sp2 or duplicates?
            # Ideally yes. For now accept.

            buf = buf.at[p].set(jnp.where(is_v, p_val, buf[p]))
            p = p + jnp.where(is_v, 1, 0)
            return (p, buf), None

        (ptr, buf), _ = jax.lax.scan(append_int, (ptr, buf), jnp.arange(clip_max))

        return (ptr, buf), None

    # Scan logic
    init = (0, out_verts)
    # We must scan static range. Assume p1 < 100.
    input_scan_len = 100

    # We mask the update logic if i >= c1
    def guarded_scan(state, i):
        new_state, _ = edge_scan(state, i)
        return (
            jax.tree_util.tree_map(
                lambda x, y: jnp.where(i < c1, x, y), new_state, state
            ),
            None,
        )

    (final_ptr, final_buf), _ = jax.lax.scan(
        guarded_scan, init, jnp.arange(input_scan_len)
    )

    return final_buf, final_ptr


def _is_point_in_polygon(
    point: jnp.ndarray, vertices: jnp.ndarray, count: int
) -> jnp.ndarray:
    """Checks if a point is inside a general polygon using Ray Casting.

    Args:
        point: Point (2,).
        vertices: Polygon vertices (N, 2).
        count: Number of valid vertices.

    Returns:
        Boolean (True if inside).
    """
    x, y = point

    def body(i, inside):
        # Ray casting algorithm
        # Check edge (p1, p2)
        idx1 = i
        idx2 = (i + 1) % count

        p1 = vertices[idx1]
        p2 = vertices[idx2]

        x1, y1 = p1
        x2, y2 = p2

        # Ray casting logic:
        # Check if ray crosses edge (p1, p2) and intersection is to the right of x.

        on_opposite_sides = (y1 > y) != (y2 > y)

        crosses = on_opposite_sides & (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1)

        return inside ^ crosses

    # Helper scan loop - but count is dynamic-ish?
    # Actually count is scalar.
    # We can iterate max_vertices and mask.

    max_v = vertices.shape[0]

    def step(inside, i):
        # Do logic only if i < count
        res = body(i, inside)
        return jnp.where(i < count, res, inside), None

    final_inside, _ = jax.lax.scan(step, False, jnp.arange(max_v))
    return final_inside


def _is_inside(point: jnp.ndarray, cp1: jnp.ndarray, cp2: jnp.ndarray) -> jnp.ndarray:
    """Checks if 'point' is inside the edge (cp1, cp2) of the clip polygon.

    Assumptions:
        - Clip polygon is convex and counter-clockwise winding.
        - 'Inside' is defined as to the left of the vector cp1->cp2.

    Args:
        point: The point to check (2,).
        cp1: The start of the clip edge (2,).
        cp2: The end of the clip edge (2,).

    Returns:
        Boolean indicating if the point is to the left of vector cp1->cp2.
    """
    return _cross_product(cp1, cp2, point) >= 0


def _line_intersection(
    p1: jnp.ndarray, p2: jnp.ndarray, p3: jnp.ndarray, p4: jnp.ndarray
) -> jnp.ndarray:
    """Finds the intersection point of line segment p1-p2 and line segment p3-p4.

    Assumptions:
        - Lines are not parallel.
        - Inputs are single points (2,), not batched.

    Args:
        p1: Start point of first line (2,).
        p2: End point of first line (2,).
        p3: Start point of second line (2,).
        p4: End point of second line (2,).

    Returns:
        The intersection point (2,).
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    # Avoid division by zero
    denom = jnp.where(jnp.abs(denom) < 1e-6, 1e-6, denom)

    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom

    x = x1 + ua * (x2 - x1)
    y = y1 + ua * (y2 - y1)

    return jnp.stack([x, y])


def _cross_product(o: jnp.ndarray, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Computes the 2D cross product of vectors OA and OB.

    Assumptions:
        - Inputs are 2D vectors or batches of 2D vectors.
        - The last dimension is size 2.

    Args:
        o: The origin point (..., 2).
        a: The first point (..., 2).
        b: The second point (..., 2).

    Returns:
        The cross product values (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0]).
    """
    return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - (
        a[..., 1] - o[..., 1]
    ) * (b[..., 0] - o[..., 0])


def _clean_vertices(vertices: jnp.ndarray, count: int) -> tuple[jnp.ndarray, int]:
    """Prunes coincident or redundant vertices from the polygon.

    This ensures that:
    1. Sequential duplicate vertices (zero-length edges) are removed.
    2. The redundant closing vertex (Last == First) is removed if present.
    3. The valid vertex count is clamped to the array size.

    Args:
        vertices: Input vertices of shape (N, 2).
        count: Number of valid vertices.

    Returns:
        A tuple (cleaned_vertices, new_count).
    """
    max_v = vertices.shape[0]

    # Clamp count to actual data size
    count = jnp.minimum(count, max_v)

    indices = jnp.arange(max_v)

    # Determine next indices (wrapping)
    next_indices = jnp.where(indices + 1 >= count, 0, indices + 1)

    curr_v = vertices
    next_v = vertices[next_indices]

    dists = jnp.linalg.norm(next_v - curr_v, axis=1)

    # Keep if distance is significant AND we are within count
    keep = (dists > 1e-6) & (indices < count)

    # Compress
    out_verts = jnp.zeros_like(vertices)
    out_ptr = 0

    def compress_step(state, i):
        buf, ptr = state
        k = keep[i]
        v = vertices[i]

        buf = buf.at[ptr].set(jnp.where(k, v, buf[ptr]))
        ptr = ptr + jnp.where(k, 1, 0)
        return (buf, ptr), None

    (temp_verts, temp_count), _ = jax.lax.scan(
        compress_step, (out_verts, out_ptr), indices
    )

    # Check closure on the CLEANED vertices
    last_idx = jnp.maximum(0, temp_count - 1)
    first_v = temp_verts[0]
    last_v = temp_verts[last_idx]

    closure_dist = jnp.linalg.norm(last_v - first_v)

    # Valid redundant closure remove:
    remove_last = (closure_dist < 1e-6) & (temp_count > 1)

    final_count = jnp.where(remove_last, temp_count - 1, temp_count)

    return temp_verts, final_count


def _buffer(
    vertices: jnp.ndarray, count: int, distance: float, max_vertices: int
) -> tuple[jnp.ndarray, int]:
    """Core buffer implementation using vertex offset and segment stitching.

    Args:
        vertices: Input vertices of shape (N, 2).
        count: Number of valid vertices.
        distance: Buffer amount (+ for dilation, - for erosion).
        max_vertices: Size of the output buffer.

    Returns:
        A tuple (buffered_vertices, valid_count).
    """

    # 1. Clean Vertices (remove coincident and redundant closure)
    vertices, count = _clean_vertices(vertices, count)

    # 2. Compute Normals for all edges
    indices = jnp.arange(vertices.shape[0])

    # Wrapping indices
    prev_indices = jnp.where(indices == 0, count - 1, indices - 1)
    next_indices = jnp.where(indices == count - 1, 0, indices + 1)

    # Allow OOB read (masked later)
    safe_prev = jnp.where(indices < count, prev_indices, 0)
    safe_next = jnp.where(indices < count, next_indices, 0)

    curr_v = vertices
    prev_v = vertices[safe_prev]
    next_v = vertices[safe_next]

    def get_normal(a, b):
        vec = b - a
        length = jnp.linalg.norm(vec)
        # Normal (y, -x) for CCW
        length = jnp.where(length < 1e-9, 1.0, length)
        return jnp.array([vec[1], -vec[0]]) / length

    n1 = jax.vmap(get_normal)(prev_v, curr_v)  # Normal of incoming edge
    n2 = jax.vmap(get_normal)(curr_v, next_v)  # Normal of outgoing edge

    # 3. Determine Convexity
    v1 = curr_v - prev_v
    v2 = next_v - curr_v
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]

    # Cross product > 0 => Left Turn => Convex (for CCW)
    is_convex = cross >= -1e-9

    verts_per_corner = 16

    def process_corner(p, n_prev, n_next, convex, d):
        return _offset_vertex(p, n_prev, n_next, convex, d, verts_per_corner)

    # generated_chunks: (MaxIn, 16, 2)
    # chunk_counts: (MaxIn,)
    corner_poly_fn = jax.vmap(process_corner, in_axes=(0, 0, 0, 0, None))
    generated_chunks, chunk_counts = corner_poly_fn(curr_v, n1, n2, is_convex, distance)

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
        vertices, count, generated_chunks, chunk_counts
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
    p: jnp.ndarray,
    n1: jnp.ndarray,
    n2: jnp.ndarray,
    is_convex: bool,
    dist: float,
    max_pts: int,
) -> tuple[jnp.ndarray, int]:
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
