import jax
import jax.numpy as jnp
import pytest
from geometry import Polygon, Circle
import ops


class TestOverflow:
    def test_buffer_overflow(self):
        # Create a circle with many edges
        circle = Circle(0, 0, 1.0, num_edges=50)  # 50 vertices

        # Buffer it with a VERY small max_vertices to force overflow
        # Buffer usually increases vertex count significantly (especially with round joins)
        # Even with miter, it might expand.
        # Let's use max_vertices=10 (smaller than input count 50)

        # ops.buffer cleans vertices first.

        res = ops.buffer(circle, 0.1, max_vertices=10)

        # Expect overflow flag to be True
        assert res.overflow

        # Expect count to be clamped to max_vertices
        assert res.count == 10

        # Result vertices shape
        assert res.vertices.shape == (10, 2)

    def test_intersection_overflow(self):
        # Create two overlapping squares
        p1 = Polygon(jnp.array([[0, 0], [2, 0], [2, 2], [0, 2]]), 4)
        p2 = Polygon(jnp.array([[1, 1], [3, 1], [3, 3], [1, 3]]), 4)

        # Intersection is a square (4 vertices).
        # Force max_vertices=2

        res = ops.intersection(p1, p2, max_vertices=2)

        assert res.overflow
        # Count can be low if clamping removed intersecting geometry
        assert res.count <= 2

    def test_union_overflow(self):
        p1 = Polygon(jnp.array([[0, 0], [2, 0], [2, 2], [0, 2]]), 4)
        p2 = Polygon(jnp.array([[1, 1], [3, 1], [3, 3], [1, 3]]), 4)

        # Union has 8 vertices roughly.
        res = ops.union(p1, p2, max_vertices=3)

        assert res.overflow
        assert res.count == 3
