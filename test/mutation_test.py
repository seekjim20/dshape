import jax
import jax.numpy as jnp
import pytest
import geometry
from src import mutation as mut_ops
from src import core  # For _is_point_in_polygon usage in tests


class TestOpsBase:
    def create_L_shape(self, origin_x=0.0, origin_y=0.0):
        vertices = jnp.array(
            [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]],
            dtype=jnp.float32,
        )
        vertices = vertices + jnp.array([origin_x, origin_y])
        return geometry.Polygon(vertices=vertices, count=6)


class TestBuffer(TestOpsBase):
    def test_buffer_square_positive(self):
        rect = geometry.Rectangle(-1, -1, 2, 2)
        dist = 0.5
        buffered = mut_ops.buffer(rect, dist)
        area = buffered.area
        assert area > 4.0
        assert jnp.abs(area - 8.785) < 0.2
        assert not buffered.self_intersect

    def test_buffer_square_negative(self):
        rect = geometry.Rectangle(-1, -1, 2, 2)
        dist = -0.5
        buffered = mut_ops.buffer(rect, dist)
        area = buffered.area
        assert area < 4.0
        assert jnp.abs(area - 1.0) < 0.1
        assert not buffered.self_intersect

    def test_buffer_l_shape(self):
        poly = self.create_L_shape()

        dist = 0.2
        buffered = mut_ops.buffer(poly, dist)
        assert buffered.count > 6
        assert buffered.area > poly.area

        dist_neg = -0.2
        eroded = mut_ops.buffer(poly, dist_neg)
        assert eroded.area < poly.area
        eroded = mut_ops.buffer(poly, dist_neg)
        assert eroded.area < poly.area
        assert not buffered.self_intersect
        assert not eroded.self_intersect

    def test_buffer_circle(self):
        c = geometry.Circle(0, 0, 1, num_edges=32)
        dist = 0.5
        buffered = mut_ops.buffer(c, dist)
        expected_area = jnp.pi * (1.5**2)
        area = buffered.area
        assert jnp.abs(area - expected_area) < 0.2
        assert not buffered.self_intersect

    def test_large_negative_buffer(self):
        rect = geometry.Rectangle(-1, -1, 2, 2)
        dist = -2.0
        buffered = mut_ops.buffer(rect, dist)
        area = buffered.area
        # With topo/inversion check, this should now return Empty (area 0) or negligible artifacts.
        assert jnp.abs(area) < 1e-6
        # assert buffered.count == 0  # Robust buffer might return non-zero points but zero area
        assert not buffered.self_intersect

    def test_buffer_concave_erosion(self):
        vertices = jnp.array(
            [[0.0, 0.0], [1.5, 0.0], [1.5, 0.5], [1.0, 0.5], [1.0, 2.0], [0.0, 2.0]],
            dtype=jnp.float32,
        )
        polygon = geometry.Polygon(vertices=vertices, count=6)
        dist = -0.1
        buffered = mut_ops.buffer(polygon, dist)

        assert buffered.area > 0
        assert jnp.isfinite(buffered.area)
        # Check that area decreased but not to zero
        assert buffered.area < polygon.area
        assert buffered.area > 1.0
        assert not buffered.self_intersect

    def test_buffer_large_erosion(self):
        # The case that used to vanish
        vertices = jnp.array(
            [[0.0, 0.0], [1.5, 0.0], [1.5, 0.5], [1.0, 0.5], [1.0, 2.0], [0.0, 2.0]],
            dtype=jnp.float32,
        )
        polygon = geometry.Polygon(vertices=vertices, count=6)
        dist = -0.3
        buffered = mut_ops.buffer(polygon, dist)

        assert buffered.area > 0.3
        assert jnp.isfinite(buffered.area)
        assert buffered.count > 3
        # Ensure it didn't collapse to 0
        assert buffered.area < polygon.area
        assert not buffered.self_intersect

    def test_buffer_degenerate_triangle(self):
        # User reported case: Triangle with count=6 (3 valid, 3 degenerate/zero-length)
        # Verify stitcher handles this gracefully.
        vertices = jnp.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            dtype=jnp.float32,
        )
        polygon = geometry.Polygon(vertices=vertices, count=6)
        dist = 0.02
        buffered = mut_ops.buffer(polygon, dist)

        assert buffered.area > 0.5
        assert jnp.abs(buffered.area - 0.57) < 0.1
        assert not buffered.self_intersect

    def test_buffer_connected_triangles(self):
        # Regression test for "Keep Loop vs Keep Head" heuristic failure.
        # Two connected triangles (bowtie/hourglass) connected at [1,0].
        # Heuristic must correctly choose the main body (Area) over the high-vertex connection sausage.
        vertices = jnp.array(
            [
                [2.0, 0.0],
                [1.0, 1.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=jnp.float32,
        )

        p1 = geometry.Polygon(vertices=vertices, count=vertices.shape[0])
        p2 = mut_ops.buffer(p1, 0.2)

        # Verify the top point [1.0, 1.0] is covered
        is_in = core._is_point_in_polygon(jnp.array([1.0, 1.0]), p2.vertices, p2.count)
        assert is_in
        assert not p2.self_intersect


class TestOffset(TestOpsBase):
    def test_offset_rectangle(self):
        rect = geometry.Rectangle(0.0, 0.0, 1.0, 1.0)
        dx, dy = 1.0, 0.5
        shifted = mut_ops.offset(rect, dx, dy)

        assert jnp.abs(shifted.area - rect.area) < 1e-5
        # Check center/vertex
        expected_v0 = jnp.array([1.0, 0.5])
        assert jnp.allclose(shifted.vertices[0], expected_v0)
        assert not shifted.self_intersect

    def test_offset_gradient(self):
        rect = geometry.Rectangle(0.0, 0.0, 1.0, 1.0)

        def loss(d):
            shifted = mut_ops.offset(rect, d[0], d[1])
            # Minimize distance to origin of first vertex
            v0 = shifted.vertices[0]
            return jnp.sum(v0**2)

        grad_fn = jax.grad(loss)
        d = jnp.array([1.0, 1.0])
        grads = grad_fn(d)

        # v0 = [0+dx, 0+dy] = [dx, dy]
        # loss = dx^2 + dy^2
        # grad = [2dx, 2dy] = [2, 2]
        assert jnp.allclose(grads, jnp.array([2.0, 2.0]))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
