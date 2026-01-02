"""Differentiable geometry primitives.

This module defines the core data structures for differentiable geometry,
specifically the Polygon class which is a JAX PyTree.
"""

import jax
import jax.numpy as jnp
from functools import partial
import dataclasses
import matplotlib.pyplot as plt
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
        count: Integer or scalar array indicating the number of valid vertices.
        overflow: Boolean indicating if the logical vertex count exceeded the buffer size.
    """

    vertices: ArrayLike
    count: ArrayLike
    overflow: bool = False

    def tree_flatten(self):
        # We treat overflow as metadata (auxiliary) usually, but since it might be
        # produced by JIT-ed code (as a boolean tensor), it should be a child?
        # If it's a python bool, it's aux. If it's a tracer/array, it's child.
        # To be safe for JIT, we treat it as a child if it can be a tracer.
        # However, usually boolean flags are better as children to allow gradients/tracing?
        # Boolean is not differentiable, but it is traced.
        return ((self.vertices, self.count, self.overflow), None)

    @classmethod
    def tree_unflatten(cls, aux, children):
        return Polygon(children[0], children[1], children[2])

    @property
    @jax.jit
    def area(self) -> Array:
        """Computes area of the polygon.

        Assumptions:
            - Vertices are ordered (CCW for positive area, CW for negative area).
            - Polygon is simple (not self-intersecting) for correct area.
            - count reflects the number of valid vertices in the beginning of the buffer.

        Returns:
            The area of the polygon.
        """
        safe_count = jnp.maximum(self.count, 1)
        indices = jnp.arange(self.vertices.shape[0])
        next_indices = (indices + 1) % safe_count

        x = self.vertices[:, 0]
        y = self.vertices[:, 1]
        x_next = x[next_indices]
        y_next = y[next_indices]

        terms = x * y_next - x_next * y

        mask = indices < self.count
        terms = terms * mask

        area_val = 0.5 * jnp.sum(terms)
        return area_val

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

        # Ensure we work with numpy arrays
        vertices_np = np.array(self.vertices)
        if isinstance(self.count, int):
            count_val = self.count
        else:
            try:
                count_val = int(self.count)
            except Exception:
                count_val = len(vertices_np)

        valid_vertices = vertices_np[:count_val]

        # Close the polygon
        if count_val > 0:
            valid_vertices = np.concatenate(
                [valid_vertices, valid_vertices[:1]], axis=0
            )

        x = valid_vertices[:, 0]
        y = valid_vertices[:, 1]

        # Use fill for polygons, but allow kwargs to control style
        # Defaults
        fill_kwargs = {"alpha": 0.5, "edgecolor": "black"}
        fill_kwargs.update(kwargs)

        ax.fill(x, y, **fill_kwargs)
        ax.set_aspect("equal")

        return ax

    @property
    def self_intersect(self) -> bool:
        """Checks if the polygon self-intersects.

        Returns:
            True if self-intersecting, False otherwise.
        """
        return _has_self_intersection(self.vertices, self.count)


@jax.tree_util.register_pytree_node_class
class Rectangle(Polygon):
    """Rectangle polygon."""

    def __init__(self, x, y, w, h):
        vertices = jnp.array(
            [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=jnp.float32
        )
        # Initialize parent Polygon
        super().__init__(vertices=vertices, count=4)


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
        super().__init__(vertices=vertices, count=num_edges)


def _has_self_intersection(vertices: ArrayLike, count: ArrayLike) -> bool:
    """Check if a polygon self-intersects.

    O(N^2) check.

    Args:
        vertices: Polygon vertices (N, 2).
        count: Number of valid vertices.

    Returns:
        True if self-intersecting.
    """
    # We generate all pairs (i, j).
    # Valid i: 0..count-1
    # Valid j: 0..count-1
    # Condition: i < j - 1 (non-adjacent)
    # Also exclude closing edge case: i=0, j=count-1.

    max_v = vertices.shape[0]

    indices = jnp.arange(max_v)

    def check_edge_i(i):
        # Edge i: (i) -> (i+1)
        # Proper indices
        idx_Next = jnp.where(i + 1 == count, 0, i + 1)  # Assumes i < count

        p1 = vertices[i]
        p2 = vertices[idx_Next]

        # Check against Edge j
        def check_edge_j(j):
            # Edge j: (j) -> (j+1)
            idx_j_Next = jnp.where(j + 1 == count, 0, j + 1)
            q1 = vertices[j]
            q2 = vertices[idx_j_Next]

            # Intersection check
            def orientation(a, b, c):
                val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
                return val

            o1 = orientation(p1, p2, q1)
            o2 = orientation(p1, p2, q2)
            o3 = orientation(q1, q2, p1)
            o4 = orientation(q1, q2, p2)

            # Strict inequality avoids endpoint touches (which are fine for adjacent edges).
            intersect = (o1 * o2 < -1e-9) & (o3 * o4 < -1e-9)

            # Valid pair logic:
            # i < count, j < count
            # Non-adjacent:
            # j != i (obviously)
            # j != i+1 (next)
            # i != j+1 (prev)
            # i != 0 or j != count-1 (closing edge vs first edge is adjacent)

            # Simple non-adjacency filter:
            # 1. i < j (symmetry) -> enforcing this avoids double check and i==j.
            # 2. j != i + 1
            # 3. not (i==0 and j==count-1)

            is_valid_pair = (i < count) & (j < count) & (i < j)
            is_adjacent = (j == i + 1) | ((i == 0) & (j == count - 1))

            should_check = is_valid_pair & (~is_adjacent)

            return intersect & should_check

        # Scan j
        row_res = jax.vmap(check_edge_j)(indices)
        return jnp.any(row_res)

    # Scan i
    total_res = jax.vmap(check_edge_i)(indices)
    return jnp.any(total_res)
