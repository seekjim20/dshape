"""Mutation operations for polygons (Buffer, Offset)."""

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from . import geometry
from . import core
from . import planarize


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
    vertices, count, ring_counts, buf_overflow = core._buffer(
        polygon.vertices, polygon.count, polygon.ring_counts, distance, max_vertices
    )

    # Note: _buffer usually cleans vertices, so checking input count > max_vertices
    # might be too aggressive if cleaning reduces it significantly.
    # But usually buffer INCREASES count. So if input > output cap, it's risky.
    # _buffer implementation handles input iteration.

    # Keep output overflow check primarily.
    # Union with buffer-internal overflow detection
    overflow = (count >= max_vertices) | buf_overflow

    # Clean the output to remove duplicates
    vertices, count = core._clean_vertices(vertices, count)

    # Robust Planarization pipeline
    # 1. Extract Edges
    # We rely on core._extract_edges to convert vertices/counts to segment soup
    # Note: _stitch_segments assumes inputs are segments.
    # core._buffer returns stitched... wait.
    # We need segments from the `vertices` result.

    # Re-extract
    segments, seg_count = core._extract_edges(
        vertices, count, ring_counts, max_vertices
    )

    # 2. Planarize
    # Max segments for planarization buffer.
    # Planarization creates O(N + K) segments.
    # Let's say 2x expansion?
    MAX_PLANAR = max_vertices * 2
    planar_segs, planar_cnt = planarize.planarize_segments(
        segments, seg_count, MAX_PLANAR
    )

    # 3. Re-Stitch
    # This forms the planar graph limit cycles.
    plan_verts, plan_cnt, plan_rcs = core._stitch_segments(
        planar_segs, planar_cnt, max_vertices, max_rings=16
    )

    # 4. Filter (for Erosion)
    # The planarized graph effectively splits self-intersecting loops into separate rings.
    # For erosion, collapsing features can form inverted (CW) loops or artifacts.
    # We apply a filter to remove these artifacts, largely based on area.

    # Check if input was single ring (ignoring implementation padding)
    input_rings = jnp.sum(polygon.ring_counts > 0)

    def positive_only_filter(v, c, rings):
        # Compute areas
        starts = jnp.cumsum(jnp.pad(rings, (1, 0))[:-1])

        def filter_ring(i):
            s = starts[i]
            rc = rings[i]

            # Extract ring verts (padded access)
            # Shoelace
            idx = jnp.arange(max_vertices)
            # Mask ring
            in_ring = (idx >= s) & (idx < s + rc)
            ring_v = jnp.where(in_ring[:, None], v, 0.0)

            x = ring_v[:, 0]
            y = ring_v[:, 1]

            # Simple circular shift inside the masked region is hard.
            # But we can assume contiguous blocks?
            # Vertices are packed by stitch.
            # So just use slice.

            # Slice dynamic? No.
            # Use rolled sum over whole array, masked?
            # If packed contiguous:
            # x[s]...x[s+rc-1]
            # area = 0.5 * sum(x_i * y_i+1 - x_i+1 * y_i)
            # For the last vertex, i+1 wraps to s.

            # Calculate cross terms
            x_next = jnp.roll(x, -1)
            y_next = jnp.roll(y, -1)
            term = x * y_next - x_next * y

            # Mask terms.
            # valid terms are s to s+rc-2.
            # Last term (s+rc-1) needs to pair with s.
            # The roll logic pairs s+rc-1 with s+rc. Which is next ring start.
            # So standard roll is wrong for packed rings.

            # Mask for "standard" edges
            std_mask = (idx >= s) & (idx < s + rc - 1)
            term_val = jnp.where(std_mask, term, 0.0)

            # Closing edge: (s+rc-1) -> s
            last_idx = s + rc - 1
            # Avoid OOB
            last_idx = jnp.minimum(last_idx, max_vertices - 1)

            x_last = v[last_idx, 0]
            y_last = v[last_idx, 1]
            x_first = v[s, 0]
            y_first = v[s, 1]

            close_term = x_last * y_first - x_first * y_last

            raw_area = 0.5 * (jnp.sum(term_val) + close_term)

            # Return Ring Count if Area > EPS, else 0 (delete)
            keep = raw_area > 1e-6
            return jnp.where(keep, rc, 0), keep

        new_rcs, keeps = jax.vmap(filter_ring)(jnp.arange(rings.shape[0]))

        # We need to repack vertices if we drop rings?
        # stitch already packed them.
        # If we drop a ring, we have gaps.
        # We should repack.

        # But for now, let's just use the `ring_counts` update.
        # However, Polygon expects packed structure?
        # `geometry.Polygon` doesn't strictly require packed, but operations like `area` do.
        # So we should repack.

        return v, c, new_rcs, keeps

    # Apply filter conditionally
    # Only if erosion (dist < 0) AND single ring input?
    is_erosion = jnp.all(distance < -1e-6)
    is_single = input_rings == 1

    should_filter = is_erosion & is_single

    # Run filter (returns possibly sparse/invalidated counts)
    filt_v, filt_c, filt_rcs, keeps = positive_only_filter(
        plan_verts, plan_cnt, plan_rcs
    )

    # Repack if filtered
    # Extract edges again? Or efficient repack?
    # extract_edges is robust.

    # Or just use the filtered ring counts?
    # If we zero out ring counts, `area` treats them as empty loops?
    # `area` property iterates `starts` based on `ring_counts`.
    # If we set ring_count=0, it skips.
    # BUT `starts` formulation `cumsum` will pack them together.
    # If data is not moved, `starts` will point to old data location?
    # Yes.
    # Ex: Rings lengths [10, 0, 10].
    # Starts: [0, 10, 10].
    # Ring 1 (len 0) range [10, 10). Empty.
    # Ring 2 (len 10) range [10, 20).
    # But Ring 2 data is actually at [10+old_len_1, ...).
    # So we MUST pack vertices if we remove rings.

    # Efficient Repack:
    # Use boolean mask of VALID vertices?
    # mask = valid_ring_indices
    # new_v = v[mask]

    # Construct mask:
    # vmap over rings to generate mask?
    def get_ring_mask(i):
        s = jnp.cumsum(jnp.pad(plan_rcs, (1, 0))[:-1])[i]
        rc = plan_rcs[i]
        kp = keeps[i]
        idx = jnp.arange(max_vertices)
        # Keep if ring kept AND in range
        return kp & (idx >= s) & (idx < s + rc)

    ring_masks = jax.vmap(get_ring_mask)(jnp.arange(16))
    total_mask = jnp.any(ring_masks, axis=0)  # Union

    # Compress
    # (max_v, 2)
    # Scan to find new positions?
    # Or just sorting trick:
    # sort by (not mask).

    sort_indices = jnp.argsort(~total_mask)
    packed_verts = filt_v[sort_indices]

    # Update count
    packed_count = jnp.sum(total_mask)

    # Update ring_counts (remove zeros)
    # sort keeps > 0
    rc_mask = filt_rcs > 0
    sort_rc = jnp.argsort(~rc_mask)
    packed_rcs = filt_rcs[sort_rc]
    # Zero out the rest
    # (implied by sort of inputs where masked are 0 anyway?)
    # Ensure tail is clean

    final_v = jax.lax.cond(should_filter, lambda: packed_verts, lambda: plan_verts)
    final_c = jax.lax.cond(should_filter, lambda: packed_count, lambda: plan_cnt)
    final_r = jax.lax.cond(should_filter, lambda: packed_rcs, lambda: plan_rcs)

    safe_count = jnp.minimum(final_c, max_vertices)

    return geometry.Polygon(
        vertices=final_v, count=safe_count, ring_counts=final_r, overflow=overflow
    )


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
        ring_counts=polygon.ring_counts,
        overflow=polygon.overflow,
    )


def rotate(
    polygon: geometry.Polygon, angle_rad: ArrayLike, center: ArrayLike
) -> geometry.Polygon:
    """Rotates the polygon by an angle around a center point.

    Args:
        polygon: Input polygon.
        angle_rad: Rotation angle in radians.
        center: Center of rotation (x, y).

    Returns:
        Rotated polygon.
    """
    center = jnp.array(center)
    vertices = polygon.vertices

    # Translate to origin
    vertices_centered = vertices - center

    # Rotation matrix
    cos_a = jnp.cos(angle_rad)
    sin_a = jnp.sin(angle_rad)

    # R * v
    # x' = x cos - y sin
    # y' = x sin + y cos
    x = vertices_centered[:, 0]
    y = vertices_centered[:, 1]

    x_new = x * cos_a - y * sin_a
    y_new = x * sin_a + y * cos_a

    vertices_rotated = jnp.stack([x_new, y_new], axis=1)

    # Translate back
    vertices_final = vertices_rotated + center

    return geometry.Polygon(
        vertices=vertices_final,
        count=polygon.count,
        ring_counts=polygon.ring_counts,
        overflow=polygon.overflow,
    )


def scale(
    polygon: geometry.Polygon, factor: ArrayLike, origin: ArrayLike
) -> geometry.Polygon:
    """Scales the polygon by a factor relative to an origin.

    Args:
        polygon: Input polygon.
        factor: Scale factor. Can be scalar (uniform) or (sx, sy) (non-uniform).
        origin: Center of scaling (x, y).

    Returns:
        Scaled polygon.
    """
    origin = jnp.array(origin)
    factor = jnp.array(factor)
    vertices = polygon.vertices

    # Translate to origin
    vertices_centered = vertices - origin

    # Scale
    # vector * scalar or vector * vector (element-wise) works automatically in JAX
    vertices_scaled = vertices_centered * factor

    # Translate back
    vertices_final = vertices_scaled + origin

    return geometry.Polygon(
        vertices=vertices_final,
        count=polygon.count,
        ring_counts=polygon.ring_counts,
        overflow=polygon.overflow,
    )
