"""Planarization algorithm for self-intersecting polygons."""

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from . import core

# Constants
EPSILON = 1e-6


def planarize_segments(
    segments: ArrayLike, count: int, max_output_segments: int
) -> tuple[Array, int]:
    """Planarizes a set of line segments by splitting them at intersections.

    Args:
        segments: Array of shape (N, 2, 2) containing segments (start, end).
        count: Number of valid segments.
        max_output_segments: Maximum size of the output buffer.

    Returns:
        A tuple (planar_segments, new_count).
        planar_segments: (max_output_segments, 2, 2)
        new_count: Number of valid output segments.
    """
    # 1. Find Intersections (N^2)
    # Returns (N, N, 2) intersection points + mask
    # Actually we want (N, K) split points per segment?

    # Let's find all t-values for each segment.
    # intersection(seg_i, seg_j) -> t_i, t_j

    # We can simplify: just collect all (N, N) potential split points.

    intersections, t_values, valid_mask = _find_all_intersections(segments, count)

    # 2. Split Segments
    # For each segment i, we have a list of t_values (from j).
    # Sort t values.
    # Generate new segments.
    # Flatten result.

    new_segs, new_cnt = _split_segments(
        segments, count, t_values, valid_mask, max_output_segments
    )

    return new_segs, new_cnt


def _cross_product(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _find_all_intersections(segments: ArrayLike, count: int):
    """Finds all pairwise intersections."""
    N = segments.shape[0]

    # P + t*R = Q + u*S
    # (P-Q) + t*R = u*S
    # Cross with S: (P-Q)xS + t(RxS) = 0
    # t = (Q-P)xS / (RxS)
    # u = (Q-P)xR / (RxS)

    def intersect_pair(i, j):
        # Determine if valid pair
        is_valid_pair = (i < count) & (j < count) & (i != j)

        # Segments
        p = segments[i, 0]
        r = segments[i, 1] - p

        q = segments[j, 0]
        s = segments[j, 1] - q

        r_cross_s = _cross_product(r, s)
        q_minus_p = q - p

        # Check parallel (cross ~ 0)
        is_parallel = jnp.abs(r_cross_s) < EPSILON

        # t and u
        t = _cross_product(q_minus_p, s) / jnp.where(is_parallel, 1.0, r_cross_s)
        u = _cross_product(q_minus_p, r) / jnp.where(is_parallel, 1.0, r_cross_s)

        # Strict intersection?
        # We generally want strict interior intersections to split.
        # If segments touch at endpoints (t=0,1 or u=0,1), they are already connected.
        # We only care if 0 < t < 1.

        strict_t = (t > EPSILON) & (t < 1.0 - EPSILON)
        strict_u = (u > EPSILON) & (u < 1.0 - EPSILON)

        has_intersect = is_valid_pair & (~is_parallel) & strict_t & strict_u

        # Calculate point (redundant if we just use t for splitting)
        # pt = p + t * r

        return t, has_intersect

    # vmap over all pairs
    # (N, N)
    t_matrix, mask_matrix = jax.vmap(
        lambda i: jax.vmap(lambda j: intersect_pair(i, j))(jnp.arange(N))
    )(jnp.arange(N))

    return None, t_matrix, mask_matrix


def _split_segments(segments, count, t_matrix, mask_matrix, max_out):
    """Splits segments based on t-values."""
    N = segments.shape[0]

    # For each segment i, we have a row of t-values t_matrix[i, :] masked by mask_matrix[i, :]
    # We want to collect them, sort them (0 < t1 < t2 ... < 1), and create sub-segments.

    # Max splits? N-1 potential intersections per segment.
    # Collecting and sorting variable length is hard in JAX.
    # But N is usually small-ish (256/512?).
    # Sort is fast.

    # Pad t_matrix invalid entries with 2.0 (so they sort on right) or -1.0.
    # We want valid t in [0,1].
    # Use 2.0 for invalid.

    t_safe = jnp.where(mask_matrix, t_matrix, 2.0)

    # Add 0.0 and 1.0 explicit endpoints to every row?
    # Or just sort, filter < 1.0.
    # Construct [0.0, sorted_ts..., 1.0]

    def process_segment(i):
        # Get ts
        ts = t_safe[i]  # (N,)

        # Sort
        ts_sorted = jnp.sort(ts)

        # We need to form intervals.
        # [0, t1], [t1, t2], ... [tn, 1]

        # Count splits?
        # Valid splits are those < 1.0 + EPS?
        valid_split_mask = ts_sorted < 1.5  # 2.0 was sentinel

        # Add 0 and 1
        # Concatenate: [0.0, ts..., 1.0]
        # But fixed size needed?
        # Output array size = N + 2?

        # If N=256, sorted is 256.
        # We produce at most N+1 segments.
        # We need a fixed buffer for sub-segments of this segment.
        # Let's say max_splits = 16? Or dynamic?
        # If N is large, this is huge.

        # Optimization: Only top K splits?
        # Robustness requires all.
        # With buffer size 256, N=256, N splits is possible.
        # Just use N+2 size.

        # Create full list of points: 0, t1, t2... 1
        # P_start = seg[i].start
        # P_vec = vector

        p = segments[i, 0]
        v = segments[i, 1] - p

        # t_points: (N+2)
        # [0.0, ...ts..., 1.0, 2.0, 2.0...]
        # We construct effective t array.

        # Better:
        # combined = [0.0] + valid_ts + [1.0]
        # BUT JAX concatenation needs static shapes.

        # Workaround:
        # Prepend 0.0 to sorted.
        # ts_sorted is [t1, t2, ..., 2.0, 2.0]
        # Shifted: [0.0, t1, t2...]
        # We want pairs (current, next).

        # Let's clean ts_sorted.
        # Replace 2.0 with 1.0?
        # If we replace all invalids with 1.0.
        # Then unique?

        ts_clamped = jnp.where(valid_split_mask, ts_sorted, 1.0)
        # Now we have [t1, t2, ..., tn, 1.0, 1.0, 1.0]
        # We need to insert 0.0 at start.

        # (N+1,)
        points_t = jnp.concatenate([jnp.array([0.0]), ts_clamped])

        # We need unique points to avoid zero-length segments.
        # Zero-length segments are typically filtered later, but basic check is good.

        # Compute points
        # points (N+1, 2)
        # p_k = p + t_k * v
        pts_out = p + points_t[:, None] * v[None, :]

        # Form segments (pts_out[k], pts_out[k+1])
        # N output segments.
        # (N, 2, 2)

        sub_starts = pts_out[:-1]
        sub_ends = pts_out[1:]

        sub_segs = jnp.stack([sub_starts, sub_ends], axis=1)

        # Determine validity
        # Valid if length > 0 (t_next > t_curr)
        # AND t_curr < 1.0 (handled by clamping invalids to 1.0)

        t_starts = points_t[:-1]
        t_ends = points_t[1:]

        # Valid: diff > epsilon
        is_valid_sub = (t_ends - t_starts) > EPSILON

        # Also check original segment valid
        is_valid_sub = is_valid_sub & (i < count)

        return sub_segs, is_valid_sub

    # vmap process
    # (N, N, 2, 2)
    all_sub_segs, all_valid_masks = jax.vmap(process_segment)(jnp.arange(N))

    # Flatten
    # (N*N, 2, 2)
    flat_segs = all_sub_segs.reshape(-1, 2, 2)
    flat_mask = all_valid_masks.reshape(-1)

    # Compact / Cap to max_out
    # We use a scan to compact? Or just sort by mask?
    # Sorting boolean mask puts Trues at end (False < True).
    # We want Trues at start.
    # Sort descending.

    sort_idx = jnp.argsort(flat_mask)[::-1]
    sorted_segs = flat_segs[sort_idx]
    sorted_mask = flat_mask[sort_idx]

    total_valid = jnp.sum(sorted_mask)

    # Take top max_out
    final_segs = sorted_segs[:max_out]
    final_mask = sorted_mask[:max_out]  # Implicitly used by count

    final_count = jnp.minimum(total_valid, max_out)

    return final_segs, final_count
