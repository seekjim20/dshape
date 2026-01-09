"""Mutation operations for polygons (Buffer, Offset)."""

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from . import geometry
from . import core
from . import planarize


@jax.jit(static_argnames=["max_vertices", "resolution"])
def buffer(
    polygon: geometry.Polygon,
    distance: ArrayLike,
    max_vertices: int = 256,
    resolution: int = 60,
) -> geometry.Polygon:
    """Computes the buffer of a polygon.

    Args:
        polygon: Input polygon.
        distance: Buffer distance (positive for dilation, negative for erosion).
        max_vertices: Size of output buffer.
        resolution: Resolution of arc segments (segments per full circle).

    Returns:
        Buffered polygon.

    Warning:
        If the result requires more than `max_vertices`, the polygon will be truncated
        and the `overflow` flag will be set to `True`. Check `result.overflow`
        and retry with a larger `max_vertices` if necessary.
    """
    vertices, count, ring_counts, buf_overflow = core._buffer(
        polygon.vertices,
        polygon.count,
        polygon.ring_counts,
        distance,
        max_vertices,
        resolution=resolution,
    )

    overflow = (count >= max_vertices) | buf_overflow

    # Clean the output to remove duplicates
    vertices, count = core._clean_vertices(vertices, count)

    # Robust Planarization pipeline
    # 1. Extract Edges
    segments, seg_count = core._extract_edges(
        vertices, count, ring_counts, max_vertices
    )

    # 2. Planarize
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
    # For erosion, filtering out negative area artifacts (inverted loops) helps.

    input_rings = jnp.sum(polygon.ring_counts > 0)

    def positive_only_filter(v, c, rings):
        # Compute areas
        starts = jnp.cumsum(jnp.pad(rings, (1, 0))[:-1])

        def filter_ring(i):
            s = starts[i]
            rc = rings[i]

            idx = jnp.arange(max_vertices)
            # Mask ring
            in_ring = (idx >= s) & (idx < s + rc)
            ring_v = jnp.where(in_ring[:, None], v, 0.0)

            x = ring_v[:, 0]
            y = ring_v[:, 1]

            # Calculate cross terms
            x_next = jnp.roll(x, -1)
            y_next = jnp.roll(y, -1)
            term = x * y_next - x_next * y

            # Mask terms.
            # valid terms are s to s+rc-2.
            std_mask = (idx >= s) & (idx < s + rc - 1)
            term_val = jnp.where(std_mask, term, 0.0)

            # Closing edge: (s+rc-1) -> s
            last_idx = s + rc - 1
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

    # Repack filtered vertices
    def get_ring_mask(i):
        s = jnp.cumsum(jnp.pad(plan_rcs, (1, 0))[:-1])[i]
        rc = plan_rcs[i]
        kp = keeps[i]
        idx = jnp.arange(max_vertices)
        # Keep if ring kept AND in range
        return kp & (idx >= s) & (idx < s + rc)

    ring_masks = jax.vmap(get_ring_mask)(jnp.arange(16))
    total_mask = jnp.any(ring_masks, axis=0)  # Union

    # Compress vertices using sort
    sort_indices = jnp.argsort(~total_mask)
    packed_verts = filt_v[sort_indices]

    # Update count
    packed_count = jnp.sum(total_mask)

    # Update ring_counts (remove zeros)
    rc_mask = filt_rcs > 0
    sort_rc = jnp.argsort(~rc_mask)
    packed_rcs = filt_rcs[sort_rc]

    final_v = jax.lax.cond(should_filter, lambda: packed_verts, lambda: plan_verts)
    final_c = jax.lax.cond(should_filter, lambda: packed_count, lambda: plan_cnt)
    final_r = jax.lax.cond(should_filter, lambda: packed_rcs, lambda: plan_rcs)

    safe_count = jnp.minimum(final_c, max_vertices)

    return geometry.Polygon(
        vertices=final_v, count=safe_count, ring_counts=final_r, overflow=overflow
    )


def translate(polygon: geometry.Polygon, dxy: ArrayLike) -> geometry.Polygon:
    """Translates the polygon by (dx, dy).

    Args:
        polygon: Input polygon.
        dxy: Translation vector (dx, dy).

    Returns:
        Offset polygon.
    """
    dxy = jnp.array(dxy)
    return geometry.Polygon(
        vertices=polygon.vertices + dxy,
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
