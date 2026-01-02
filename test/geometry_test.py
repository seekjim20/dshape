import jax
import jax.numpy as jnp
import numpy as np
import pytest
import geometry


class TestPolygon:
    def create_L_shape(self, origin_x=0.0, origin_y=0.0):
        # L-shape 2x2 with 1x1 removed from top-right
        # (0,0) -> (2,0) -> (2,1) -> (1,1) -> (1,2) -> (0,2)
        vertices = jnp.array(
            [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]],
            dtype=jnp.float32,
        )
        vertices = vertices + jnp.array([origin_x, origin_y])
        return geometry.Polygon(vertices=vertices, count=6)

    def test_ccw_triangle(self):
        # Counter-Clockwise Triangle (Expected Positive Area)
        vertices = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
        poly = geometry.Polygon(vertices=vertices, count=3)
        area = poly.area
        assert jnp.abs(area - 0.5) < 1e-6
        assert area > 0

    def test_cw_triangle(self):
        # Clockwise Triangle (Expected Negative Area)
        vertices = jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)
        poly = geometry.Polygon(vertices=vertices, count=3)
        area = poly.area
        assert jnp.abs(area - (-0.5)) < 1e-6
        assert area < 0

    def test_concave_area(self):
        l_shape = self.create_L_shape()
        area = l_shape.area
        expected_area = 3.0
        assert jnp.abs(area - expected_area) < 1e-4


class TestRectangle:
    def test_rectangle_area(self):
        rect = geometry.Rectangle(0, 0, 2, 2)
        area = rect.area
        assert jnp.abs(area - 4.0) < 1e-6
        assert area > 0

    def test_rectangle_reversed(self):
        rect = geometry.Rectangle(0, 0, 2, 2)
        rev_vertices = rect.vertices[::-1]
        rev_poly = geometry.Polygon(vertices=rev_vertices, count=4)
        area = rev_poly.area
        assert jnp.abs(area - (-4.0)) < 1e-6
        assert area < 0


class TestCircle:
    def test_circle_creation(self):
        x, y, r = 0.0, 0.0, 1.0
        num_edges = 100
        circle = geometry.Circle(x, y, r, num_edges)

        assert circle.count == num_edges
        assert circle.vertices.shape[0] == num_edges

        expected_area = 0.5 * num_edges * r**2 * np.sin(2 * np.pi / num_edges)
        area = circle.area
        assert jnp.abs(area - expected_area) < 1e-4

    def test_circle_jit(self):
        x, y, r = 0.0, 0.0, 1.0
        circle = geometry.Circle(x, y, r, num_edges=32)

        @jax.jit
        def area_fn(p):
            return p.area

        res = area_fn(circle)
        assert res > 3.0

    def test_circle_in_trace(self):
        @jax.jit
        def f(c):
            return c.area

        circle = geometry.Circle(0, 0, 1)
        res = f(circle)
        assert res > 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
