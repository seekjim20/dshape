"""Differentiable geometry primitives.

This module defines the core data structures for differentiable geometry,
specifically the Polygon class which is a JAX PyTree.
"""

import jax
import jax.numpy as jnp
from functools import partial
import dataclasses
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import numpy as np


# type hints
from jax import Array
from jax.typing import ArrayLike


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class Polygon:
    """Class representing a polygon with a maximum vertex buffer.

    Attributes:
        vertices: Array of shape (max_vertices, 2) containing vertex coordinates.
        count: Integer or scalar array indicating the total number of valid vertices.
        ring_counts: Array of shape (max_rings,) indicating the number of vertices in each ring.
        overflow: Boolean indicating if the logical vertex count exceeded the buffer size.
    """

    vertices: ArrayLike
    count: ArrayLike | None = None
    ring_counts: ArrayLike | None = None
    overflow: bool = False

    def __post_init__(self):
        if self.count is None:
            self.count = self.vertices.shape[0]
        if self.ring_counts is None:
            # Default to single ring containing all vertices
            # We treat scalar count as single ring
            # If ring_counts is missing, we create a default 1-element array
            self.ring_counts = jnp.array([self.count], dtype=jnp.int32)

    def tree_flatten(self):
        return ((self.vertices, self.count, self.ring_counts, self.overflow), None)

    @classmethod
    def tree_unflatten(cls, aux, children):
        return Polygon(children[0], children[1], children[2], children[3])

    @property
    @jax.jit
    def area(self) -> Array:
        """Computes area of the polygon (sum of signed areas of all rings).

        Assumptions:
            - Vertices are ordered (CCW for positive area, CW for negative area).
            - Exterior rings should be CCW (positive), Interior (holes) CW (negative).
        """
        # We need to iterate over rings.
        max_rings = self.ring_counts.shape[0]
        max_v = self.vertices.shape[0]

        # Calculate start indices for each ring
        starts = jnp.cumsum(jnp.pad(self.ring_counts, (1, 0))[:-1])

        def ring_area(i):
            c = self.ring_counts[i]
            start = starts[i]

            # Indices relative to start
            ind_rel = jnp.arange(max_v)
            # Mask for valid vertices in this ring
            valid = ind_rel < c

            # Absolute indices
            # Need safe handling for mod: avoid mod 0 if c=0
            safe_c = jnp.maximum(c, 1)

            curr_abs = start + ind_rel
            next_rel = (ind_rel + 1) % safe_c
            next_abs = start + next_rel

            # Fetch vertices (clamp to max_v-1 to avoid OOB read even if masked out)
            curr_abs = jnp.minimum(curr_abs, max_v - 1)
            next_abs = jnp.minimum(next_abs, max_v - 1)

            x = self.vertices[curr_abs, 0]
            y = self.vertices[curr_abs, 1]
            x_next = self.vertices[next_abs, 0]
            y_next = self.vertices[next_abs, 1]

            terms = x * y_next - x_next * y
            ring_val = 0.5 * jnp.sum(terms * valid)

            return jnp.where(c > 0, ring_val, 0.0)

        # Vectorize over rings
        areas = jax.vmap(ring_area)(jnp.arange(max_rings))
        return jnp.sum(areas)

    def plot(self, ax=None, **kwargs) -> plt.Axes:
        """Plots the polygon using matplotlib.

        Args:
            ax: Optional matplotlib Axes object. If None, a new figure is created.
            **kwargs: Additional arguments passed to ax.fill (e.g., color, alpha, label).

        Returns:
            The matplotlib Axes object containing the polygon plot.
        """
        if ax is None:
            fig, ax = plt.subplots()

        # Ensure we work with numpy arrays for iteration
        vertices_np = np.array(self.vertices)

        # Handle count scalar/array
        try:
            count_val = int(self.count)
        except Exception:
            count_val = len(vertices_np)

        if count_val == 0:
            return ax

        # Build Path
        codes = []
        verts = []

        # Use ring_counts if available, else single ring
        if self.ring_counts is None:
            rcs = np.array([count_val])
        else:
            rcs = np.array(self.ring_counts)

        start = 0
        for rc in rcs:
            if rc <= 0:
                break

            length = int(rc)
            ring_v = vertices_np[start : start + length]

            # MOVETO first
            verts.append(ring_v[0])
            codes.append(Path.MOVETO)

            # LINETO others
            for i in range(1, length):
                verts.append(ring_v[i])
                codes.append(Path.LINETO)

            # CLOSEPOLY
            # For CLOSEPOLY, providing a vertex is required but ignored. We use the start vertex.
            verts.append(ring_v[0])
            codes.append(Path.CLOSEPOLY)

            start += length
            if start >= count_val:
                break

        path = Path(verts, codes)

        # Default kwargs
        patch_kwargs = {"alpha": 0.5, "facecolor": "blue", "edgecolor": "black"}

        # Update with user arguments
        patch_kwargs.update(kwargs)

        # Handle 'color' meta-arg: maps to facecolor, but we enforce edge=black (default)
        if "color" in kwargs:
            patch_kwargs["facecolor"] = kwargs["color"]
            # Remove 'color' to prevent it from overriding edgecolor in PathPatch
            # (unless user explicitly passed both, in which case we still prefer splitting)
            if "color" in patch_kwargs:
                del patch_kwargs["color"]

        # Re-enforce black edge if not explicitly provided by user
        # (This protects against cases where color might have set implied edge behavior before deletion,
        # or if we want to be very sure)
        if "edgecolor" not in kwargs:
            patch_kwargs["edgecolor"] = "black"

        patch = PathPatch(path, **patch_kwargs)
        ax.add_patch(patch)

        # Explicitly update datalim with vertices
        all_verts = np.array(verts)
        ax.update_datalim(all_verts)
        ax.autoscale_view()
        ax.set_aspect("equal")
        return ax

    @property
    def self_intersect(self) -> bool:
        """Checks if the polygon self-intersects.

        Returns:
            True if self-intersecting, False otherwise.
        """
        return _has_self_intersection(self.vertices, self.count, self.ring_counts)

    @property
    def exteriors(self) -> list["Polygon"]:
        """Returns a list of Polygons representing the exterior rings (positive area).

        Note: This method is NOT JIT-compatible as it returns a Python list.
        """
        return self._extract_rings(sign=1)

    @property
    def interiors(self) -> list["Polygon"]:
        """Returns a list of Polygons representing the interior rings (negative area).

        Note: This method is NOT JIT-compatible as it returns a Python list.
        """
        return self._extract_rings(sign=-1)

    def _extract_rings(self, sign: int) -> list["Polygon"]:
        """Helper to extract rings based on area sign."""
        # Ensure we have concrete values (move to CPU if needed)
        try:
            rc_np = np.array(self.ring_counts)
            v_np = np.array(self.vertices)
        except Exception:
            # If we are strictly inside JIT, this might fail or be slow.
            # Assuming usage outside JIT for high-level API.
            raise RuntimeError(
                "Cannot access exteriors/interiors from within JIT compilation."
            )

        polys = []
        start = 0
        max_v = v_np.shape[0]

        for c in rc_np:
            if c <= 0:
                break

            # Extract ring vertices
            end = start + c
            # Clamp end to max_v to avoid error if ring_counts implies more than buffer
            # (though that shouldn't happen in valid polys)
            safe_end = min(end, max_v)

            ring_verts = v_np[start:safe_end]

            # Compute signed area to check sign
            # Shoelace formula
            x = ring_verts[:, 0]
            y = ring_verts[:, 1]
            x_next = np.roll(x, -1)
            y_next = np.roll(y, -1)
            area = 0.5 * np.sum(x * y_next - x_next * y)

            # Check sign (allow small epsilon for float noise, though usually robust)
            if sign > 0 and area > 1e-9:
                polys.append(Polygon(vertices=jnp.array(ring_verts), count=c))
            elif sign < 0 and area < -1e-9:
                polys.append(Polygon(vertices=jnp.array(ring_verts), count=c))

            start += c
            if start >= max_v:
                break

        return polys

    @classmethod
    def from_exteriors_interiors(
        cls, exteriors: list["Polygon"], interiors: list["Polygon"] = None
    ) -> "Polygon":
        """Constructs a Polygon from explicit lists of exterior and interior rings.

        Args:
            exteriors: List of Polygons representing shells (must be CCW/positive).
            interiors: List of Polygons representing holes (must be CW/negative).
                       If they are passed as Polygons with default (positive) winding,
                       they will be REVERSED automatically.
        """
        if interiors is None:
            interiors = []

        all_polys = exteriors + interiors
        if not all_polys:
            return cls(vertices=jnp.zeros((0, 2)), count=0, ring_counts=jnp.zeros(1))

        # Collect vertices and counts
        total_verts = []
        ring_counts_list = []

        for p in exteriors:
            # Check orientation? Assume user gives CCW or we trust them.
            # We could enforce CCW for exteriors.
            v = p.vertices[: p.count]
            # Ensure positive area
            # (Optional check)
            total_verts.append(v)
            ring_counts_list.append(p.count)

        for p in interiors:
            v = p.vertices[: p.count]
            # Enforce CW (negative) for interiors
            # Check area
            # We need to compute area to know if we need to flip
            # Using numpy for this check as this is a factory method
            v_np = np.array(v)
            x = v_np[:, 0]
            y = v_np[:, 1]
            area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)

            if area > 0:
                # Flip to make it a hole
                v = jnp.flip(v, axis=0)

            total_verts.append(v)
            ring_counts_list.append(int(p.count))  # Ensure scalar

        # Stack
        all_v = jnp.concatenate(total_verts, axis=0)
        total_count = all_v.shape[0]
        r_counts = jnp.array(ring_counts_list, dtype=jnp.int32)

        return cls(vertices=all_v, count=total_count, ring_counts=r_counts)

    def __add__(self, other: "Polygon") -> "Polygon":
        """Union operator (+)"""
        import set_ops

        if not isinstance(other, Polygon):
            return NotImplemented

        return set_ops.union([self, other])

    def __sub__(self, other: "Polygon") -> "Polygon":
        """Difference operator (-)"""
        import set_ops

        if not isinstance(other, Polygon):
            return NotImplemented

        return set_ops.difference(self, other)

    def __mul__(self, other: "Polygon") -> "Polygon":
        """Intersection operator (*)"""
        import set_ops

        if not isinstance(other, Polygon):
            return NotImplemented

        return set_ops.intersection(self, other)

    @property
    def convex_hull(self) -> "Polygon":
        """Computes the convex hull of the polygon."""
        import constructive

        return constructive.convex_hull(self)


@jax.tree_util.register_pytree_node_class
class Rectangle(Polygon):
    """Rectangle polygon."""

    def __init__(self, x, y, w, h):
        vertices = jnp.array(
            [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=jnp.float32
        )
        # Initialize parent Polygon
        super().__init__(vertices=vertices, count=4, ring_counts=jnp.array([4]))


@jax.tree_util.register_pytree_node_class
class Circle(Polygon):
    """Circle polygon approximated by N edges."""

    def __init__(self, x, y, r, num_edges=32):
        # Generate angles
        theta = jnp.linspace(0, 2 * jnp.pi, num_edges, endpoint=False)

        # Stack into (N, 2) array
        vx = x + r * jnp.cos(theta)
        vy = y + r * jnp.sin(theta)
        vertices = jnp.stack([vx, vy], axis=1).astype(jnp.float32)

        # Initialize parent Polygon
        super().__init__(
            vertices=vertices, count=num_edges, ring_counts=jnp.array([num_edges])
        )


def _has_self_intersection(
    vertices: ArrayLike, count: ArrayLike, ring_counts: ArrayLike = None
) -> bool:
    """Check if a polygon self-intersects.

    O(N^2) check.

    Args:
        vertices: Polygon vertices (N, 2).
        count: Total number of valid vertices.
        ring_counts: Optional array of ring counts. If None, assumes single ring.

    Returns:
        True if self-intersecting.
    """
    max_v = vertices.shape[0]

    if ring_counts is None:
        ring_counts = jnp.array([count])

    # Calculate start indices for each ring
    starts = jnp.cumsum(jnp.pad(ring_counts, (1, 0))[:-1])
    num_rings = ring_counts.shape[0]

    indices = jnp.arange(max_v)

    def get_next(i):
        # Determine which ring i belongs to
        # ring k: starts[k] <= i < starts[k] + ring_counts[k]

        # We can scan rings to find the one.
        # Since we just need to return the next index in the ring.

        # Vectorized check better than scan if num_rings is small.
        # Assume num_rings <= max_v or logic is efficient enough.

        # Map i to ring index k
        # starts <= i -> mask
        # Last True is the ring index.
        is_after_start = i >= starts
        # We need the last true index.
        # sum - 1 gives index if they are sorted and contiguous starts?
        # Yes, starts are monotonic.
        k = jnp.sum(is_after_start) - 1

        # Clamp k valid
        k = jnp.maximum(0, jnp.minimum(k, num_rings - 1))

        s = starts[k]
        c = ring_counts[k]

        # Local index
        local = i - s

        # Next local
        # If c=0, this logic is moot but safe_c handles mod
        safe_c = jnp.maximum(c, 1)
        next_local = (local + 1) % safe_c

        return s + next_local

    idx_next_map = jax.vmap(get_next)(indices)

    def check_edge_i(i):
        p1 = vertices[i]
        p2 = vertices[idx_next_map[i]]
        valid_i = i < count

        def check_edge_j(j):
            q1 = vertices[j]
            q2 = vertices[idx_next_map[j]]
            valid_j = j < count

            def orientation(a, b, c):
                val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
                return val

            o1 = orientation(p1, p2, q1)
            o2 = orientation(p1, p2, q2)
            o3 = orientation(q1, q2, p1)
            o4 = orientation(q1, q2, p2)

            intersect = (o1 * o2 < -1e-9) & (o3 * o4 < -1e-9)

            # Adjacency check
            next_i = idx_next_map[i]
            next_j = idx_next_map[j]

            is_identity = i == j
            is_adj_1 = i == next_j
            is_adj_2 = next_i == j

            is_adjacent = is_identity | is_adj_1 | is_adj_2

            should_check = valid_i & valid_j & (i < j) & (~is_adjacent)

            return intersect & should_check

        row_res = jax.vmap(check_edge_j)(indices)
        return jnp.any(row_res)

    total_res = jax.vmap(check_edge_i)(indices)
    return jnp.any(total_res)
