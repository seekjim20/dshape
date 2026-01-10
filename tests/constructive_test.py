import jax
import jax.numpy as jnp
import pytest
from dshape import geometry
from dshape import constructive


class TestConstructive:
    def test_convex_hull_convex_identity(self):
        # Hull of a square is the square itself
        # Order might shift, but area should be same
        rect = geometry.Polygon.rectangle(0, 0, 1, 1)
        hull = constructive.convex_hull(rect)

        assert jnp.abs(hull.area - rect.area) < 1e-5
        assert hull.count == 4

    def test_convex_hull_l_shape(self):
        # L-shape: (0,0)-(2,0)-(2,1)-(1,1)-(1,2)-(0,2)
        # Hull should fill the corner: (0,0)-(2,0)-(2,1)-(1,2)-(0,2) -> Missing (1,1)
        # Resulting shape is polygon minus that concave vertex?
        # Actually it adds the diagonal (2,1)-(1,2).
        # Area: L-shape is 3. Square (0,0)-(2,2) area is 4.
        # Cut corner is triangle (2,2)-(2,1)-(1,2). area 0.5.
        # So Hull area should be 4 - 0.5 = 3.5.

        vertices = jnp.array(
            [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]],
            dtype=jnp.float32,
        )
        l_poly = geometry.Polygon(vertices=vertices, count=6)

        hull = constructive.convex_hull(l_poly)

        assert hull.area > l_poly.area
        assert jnp.abs(hull.area - 3.5) < 1e-5
        # 5 vertices
        assert hull.count == 5

    def test_convex_hull_with_holes(self):
        # Square with hole. Hull should be just the Square.
        sq_verts = jnp.array([[0, 0], [3, 0], [3, 3], [0, 3]], dtype=jnp.float32)
        hole_verts = jnp.array([[1, 1], [1, 2], [2, 2], [2, 1]], dtype=jnp.float32)

        # Combine
        combined_verts = jnp.concatenate([sq_verts, hole_verts])
        rc = jnp.array([4, 4])
        poly = geometry.Polygon(vertices=combined_verts, count=8, ring_counts=rc)

        # Area 9 - 1 = 8
        assert jnp.abs(poly.area - 8.0) < 1e-5

        hull = constructive.convex_hull(poly)

        # Hull area should be 9
        assert jnp.abs(hull.area - 9.0) < 1e-5

    def test_random_points(self):
        # Generate random cloud
        key = jax.random.PRNGKey(0)
        points = jax.random.uniform(key, (50, 2))

        poly = geometry.Polygon(vertices=points, count=50)

        hull = constructive.convex_hull(poly)

        # Verify: All original points are inside or on boundary of hull?
        # That's hard to check efficiently without point-in-polygon loop.
        # But we can check Area is invariant to adding internal points?

        # Check simple property: Hull count <= input count
        assert hull.count <= 50
        # Area > 0
        assert hull.area > 0

    def test_convex_hull_grad(self):
        # Regression test for reverse-mode differentiation compatibility
        def calculate_area(radius):
            radius = jnp.array(radius)
            # Circle constructor uses tuple (xy) and radius since refactor
            c = geometry.Polygon.circle((0.0, 0.0), radius, num_edges=50)
            hull = constructive.convex_hull(c)
            return hull.area

        # Forward
        r = 1.0
        res = calculate_area(r)
        assert res > 2.0

        # Backward
        grad_fn = jax.grad(calculate_area)
        g = grad_fn(r)

        # Exact area of N-gon circle: 0.5 * N * R^2 * sin(2pi/N)
        # Derivative wrt R: N * R * sin(2pi/N)
        # N=50, R=1, sin(2pi/50) ~ 0.125
        # g ~ 50 * 1 * 0.125 ~ 6.25
        # Approx 2*pi*R = 6.28

        expected_g = 2 * jnp.pi * r
        assert jnp.abs(g - expected_g) < 0.1


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
