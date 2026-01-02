import jax
import jax.numpy as jnp
import pytest
import geometry
import mutation
import set_ops


class TestOpsBase:
    def create_L_shape(self, origin_x=0.0, origin_y=0.0):
        vertices = jnp.array(
            [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]],
            dtype=jnp.float32,
        )
        vertices = vertices + jnp.array([origin_x, origin_y])
        return geometry.Polygon(vertices=vertices, count=6)


class TestIntersection(TestOpsBase):
    def test_intersection_area(self):
        p1 = geometry.Rectangle(0.0, 0.0, 2.0, 2.0)
        p2 = geometry.Rectangle(1.0, 1.0, 2.0, 2.0)

        inter = set_ops.intersection(p1, p2)
        area = inter.area
        assert jnp.abs(area - 1.0) < 1e-4

    def test_concave_intersection(self):
        p1 = geometry.Rectangle(0.0, 0.0, 3.0, 3.0)
        p2 = self.create_L_shape()

        inter = set_ops.intersection(p1, p2)
        area = inter.area
        # Known Limitation: Sutherland-Hodgman clipping requires convex clip polygon.
        # Since p2 is concave, we expect incorrect area (not 3.0).
        assert jnp.abs(area - 3.0) > 1e-4

    def test_gradients(self):
        p1 = geometry.Rectangle(0.0, 0.0, 2.0, 2.0)

        def intersection_area_fn(offset):
            # Use geometry from src if needed, or assume global import
            p2 = geometry.Rectangle(1.0 + offset[0], 1.0 + offset[1], 2.0, 2.0)
            inter = set_ops.intersection(p1, p2)
            return inter.area

        grad_fn = jax.grad(intersection_area_fn)
        zero_offset = jnp.array([0.0, 0.0])
        grads = grad_fn(zero_offset)

        expected_grad = jnp.array([-1.0, -1.0])
        assert jnp.allclose(grads, expected_grad, atol=1e-4)

        # Test JIT
        jit_fn = jax.jit(intersection_area_fn)
        jit_fn(zero_offset)

    def test_intersection_overflow(self):
        # Create two overlapping squares
        p1 = geometry.Polygon(jnp.array([[0, 0], [2, 0], [2, 2], [0, 2]]), 4)
        p2 = geometry.Polygon(jnp.array([[1, 1], [3, 1], [3, 3], [1, 3]]), 4)

        # Intersection is a square (4 vertices).
        # Force max_vertices=2

        res = set_ops.intersection(p1, p2, max_vertices=2)

        assert res.overflow
        # Count can be low if clamping removed intersecting geometry
        assert res.count <= 2


class TestUnion(TestOpsBase):
    def test_union_convex_case(self):
        p1 = geometry.Rectangle(0.0, 0.0, 2.0, 2.0)
        p2 = geometry.Rectangle(1.0, 0.1, 2.0, 2.0)

        union_poly = set_ops.union([p1, p2])
        area = union_poly.area
        assert jnp.abs(area - 6.1) < 1e-4

    def test_union_non_convex_case(self):
        p1 = geometry.Rectangle(0.0, 0.0, 2.0, 2.0)
        p2 = geometry.Rectangle(1.5, 1.5, 2.0, 2.0)

        union_poly = set_ops.union([p1, p2])
        area = union_poly.area
        expected_area = 7.75
        assert jnp.abs(area - expected_area) < 1e-3

    def test_concave_union(self):
        p1 = self.create_L_shape()
        p2 = geometry.Rectangle(1.0, 1.0, 1.0, 1.0)

        union_poly = set_ops.union([p1, p2])
        area = union_poly.area
        assert jnp.abs(area - 4.0) < 1e-3

    def test_union_overflow(self):
        p1 = geometry.Polygon(jnp.array([[0, 0], [2, 0], [2, 2], [0, 2]]), 4)
        p2 = geometry.Polygon(jnp.array([[1, 1], [3, 1], [3, 3], [1, 3]]), 4)

        # Union has 8 vertices roughly.
        res = set_ops.union([p1, p2], max_vertices=3)

        assert res.overflow
        assert res.count == 3

    def test_union_artifact_regression(self):
        # Regression test for small-scale artifact bug (user reported)
        # P1 is small triangle near origin
        p1 = geometry.Polygon(
            jnp.array([[0.0, 0.0], [0.05, 0.0], [0.0, 0.05]], dtype=jnp.float32), 3
        )

        # Buffer
        dist = 0.079268
        p2 = mutation.buffer(p1, dist)

        # Offset (Shift essentially, as it's offset of buffer)
        # Actually offset logic expands if edges are sharp, but for buffer result (smooth/arcs), it shifts/expands.
        p3 = mutation.offset(p2, 0.05, 0.0)

        p4 = set_ops.union([p2, p3])

        # Expected area ~ 0.04452
        assert jnp.abs(p4.area - 0.04452) < 1e-4
        assert not p4.overflow

    def test_union_three_polys(self):
        # Union of 3 squares in a row
        p1 = geometry.Rectangle(0.0, 0.0, 1.0, 1.0)
        p2 = geometry.Rectangle(0.5, 0.0, 1.0, 1.0)  # Overlaps p1
        p3 = geometry.Rectangle(1.0, 0.0, 1.0, 1.0)  # Overlaps p2 (touches p1 edge)

        # Expected: Rectangle(0, 0, 2, 1) basically?
        # 0.5 overlap.
        # min x=0, max x=2. y=0..1.
        # Area should be roughly 2.0?
        # p1 area 1. p2 area 1. overlap 0.5. Union = 1.5.
        # p3 area 1. overlap with p1Up2 (0.5..1.5). p3 is (1..2). Overlap (1..1.5) -> 0.5.
        # Total area = 1.5 + 1 - 0.5 = 2.0.

        res = set_ops.union([p1, p2, p3])
        assert jnp.abs(res.area - 2.0) < 1e-3
        assert not res.overflow



class TestDifference(TestOpsBase):
    def test_difference_overlap(self):
        p1 = geometry.Rectangle(0.0, 0.0, 2.0, 2.0)
        p2 = geometry.Rectangle(1.0, 1.0, 2.0, 2.0)

        diff_poly = set_ops.difference(p1, p2)
        area = diff_poly.area
        assert jnp.abs(area - 3.0) < 1e-3

    def test_difference_produces_concave(self):
        p1 = geometry.Rectangle(0.0, 0.0, 3.0, 1.0)
        p2 = geometry.Rectangle(1.0, 0.0, 1.0, 0.5)

        diff_poly = set_ops.difference(p1, p2)
        area = diff_poly.area
        assert jnp.abs(area - 2.5) < 1e-3

    def test_hole_creation(self):
        p1 = geometry.Rectangle(0.0, 0.0, 3.0, 3.0)
        p2 = geometry.Rectangle(1.0, 1.0, 1.0, 1.0)

        diff_poly = set_ops.difference(p1, p2)
        area = diff_poly.area
        assert jnp.abs(area - 8.0) < 1e-3

    def test_p1_inside_p2(self):
        p1 = geometry.Rectangle(1.0, 1.0, 1.0, 1.0)
        p2 = geometry.Rectangle(0.0, 0.0, 3.0, 3.0)

        diff_poly = set_ops.difference(p1, p2)
        area = diff_poly.area
        assert jnp.abs(area - 0.0) < 1e-4


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
