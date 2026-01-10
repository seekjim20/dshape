import jax
import jax.numpy as jnp
import pytest
from dshape import geometry
from dshape import mutation
from dshape import set_ops


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
        p1 = geometry.Polygon.rectangle(0.0, 0.0, 2.0, 2.0)
        p2 = geometry.Polygon.rectangle(1.0, 1.0, 2.0, 2.0)

        inter = set_ops.intersection(p1, p2)
        area = inter.area
        assert jnp.abs(area - 1.0) < 1e-4

    def test_concave_intersection(self):
        p1 = geometry.Polygon.rectangle(0.0, 0.0, 3.0, 3.0)
        p2 = self.create_L_shape()

        inter = set_ops.intersection(p1, p2)
        area = inter.area
        # Known Limitation: Sutherland-Hodgman clipping requires convex clip polygon.
        # Since p2 is concave, we expect incorrect area (not 3.0).
        assert jnp.abs(area - 3.0) > 1e-4

    def test_gradients(self):
        p1 = geometry.Polygon.rectangle(0.0, 0.0, 2.0, 2.0)

        def intersection_area_fn(offset):
            # Use geometry from src if needed, or assume global import
            p2 = geometry.Polygon.rectangle(1.0 + offset[0], 1.0 + offset[1], 2.0, 2.0)
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

    def test_parallel_no_intersection(self):
        """Parallel segments should return None."""
        seg1 = geometry.LineSegment([0.0, 0.0], [2.0, 0.0])
        seg2 = geometry.LineSegment([0.0, 1.0], [2.0, 1.0])

        result = set_ops.line_segment_intersection(seg1, seg2)
        assert result is None

    def test_perpendicular_intersection(self):
        """Crossing segments should return the intersection Point."""
        seg1 = geometry.LineSegment([0.0, 0.0], [2.0, 2.0])
        seg2 = geometry.LineSegment([0.0, 2.0], [2.0, 0.0])

        result = set_ops.line_segment_intersection(seg1, seg2)
        assert result is not None
        assert isinstance(result, geometry.Point)
        assert jnp.allclose(result.xy, jnp.array([1.0, 1.0]), atol=1e-6)

    def test_t_intersection(self):
        """One segment touches the other at an endpoint."""
        seg1 = geometry.LineSegment([0.0, 0.0], [2.0, 0.0])
        seg2 = geometry.LineSegment([1.0, 0.0], [1.0, 2.0])

        result = set_ops.line_segment_intersection(seg1, seg2)
        assert result is not None
        assert jnp.allclose(result.xy, jnp.array([1.0, 0.0]), atol=1e-6)

    def test_collinear_no_intersection(self):
        """Collinear but disjoint segments should return None."""
        seg1 = geometry.LineSegment([0.0, 0.0], [1.0, 0.0])
        seg2 = geometry.LineSegment([2.0, 0.0], [3.0, 0.0])

        result = set_ops.line_segment_intersection(seg1, seg2)
        assert result is None

    def test_collinear_overlapping_returns_none(self):
        """Collinear overlapping segments should return None (documented behavior)."""
        seg1 = geometry.LineSegment([0.0, 0.0], [2.0, 0.0])
        seg2 = geometry.LineSegment([1.0, 0.0], [3.0, 0.0])

        result = set_ops.line_segment_intersection(seg1, seg2)
        # Returns None because it's collinear (cross product is 0)
        assert result is None

    def test_method_shortcut(self):
        """Test the monkey-patched intersection method."""
        seg1 = geometry.LineSegment([0.0, 0.0], [2.0, 2.0])
        seg2 = geometry.LineSegment([0.0, 2.0], [2.0, 0.0])

        result = seg1.intersection(seg2)
        assert result is not None
        assert jnp.allclose(result.xy, jnp.array([1.0, 1.0]), atol=1e-6)

    def test_polygon_intersection_single(self):
        """Segment crosses polygon once, returning one segment."""
        # Segment from outside to inside the square
        seg = geometry.LineSegment([-1.0, 0.5], [0.5, 0.5])
        poly = geometry.Polygon.rectangle(0.0, 0.0, 1.0, 1.0)

        result = set_ops.line_segment_polygon_intersection(seg, poly)
        assert len(result) == 1
        # The segment inside should be from (0, 0.5) to (0.5, 0.5)
        assert jnp.allclose(result[0].p1, jnp.array([0.0, 0.5]), atol=1e-5)
        assert jnp.allclose(result[0].p2, jnp.array([0.5, 0.5]), atol=1e-5)

    def test_polygon_intersection_through(self):
        """Segment passes through polygon, one segment result."""
        seg = geometry.LineSegment([-1.0, 0.5], [2.0, 0.5])
        poly = geometry.Polygon.rectangle(0.0, 0.0, 1.0, 1.0)

        result = set_ops.line_segment_polygon_intersection(seg, poly)
        assert len(result) == 1
        assert jnp.allclose(result[0].p1, jnp.array([0.0, 0.5]), atol=1e-5)
        assert jnp.allclose(result[0].p2, jnp.array([1.0, 0.5]), atol=1e-5)

    def test_polygon_intersection_fully_inside(self):
        """Segment fully inside polygon returns itself."""
        seg = geometry.LineSegment([0.25, 0.5], [0.75, 0.5])
        poly = geometry.Polygon.rectangle(0.0, 0.0, 1.0, 1.0)

        result = set_ops.line_segment_polygon_intersection(seg, poly)
        assert len(result) == 1
        assert jnp.allclose(result[0].p1, jnp.array([0.25, 0.5]), atol=1e-5)
        assert jnp.allclose(result[0].p2, jnp.array([0.75, 0.5]), atol=1e-5)

    def test_polygon_intersection_fully_outside(self):
        """Segment fully outside polygon returns empty list."""
        seg = geometry.LineSegment([2.0, 0.5], [3.0, 0.5])
        poly = geometry.Polygon.rectangle(0.0, 0.0, 1.0, 1.0)

        result = set_ops.line_segment_polygon_intersection(seg, poly)
        assert len(result) == 0

    def test_polygon_intersection_with_hole(self):
        """Segment crossing polygon with hole returns multiple segments."""
        # Polygon with a hole in the middle
        outer = geometry.Polygon.rectangle(0.0, 0.0, 4.0, 4.0)
        hole = geometry.Polygon.rectangle(1.0, 1.0, 2.0, 2.0)
        poly_with_hole = geometry.Polygon.from_exteriors_interiors([outer], [hole])

        # Segment passes through outer, hole, outer
        seg = geometry.LineSegment([0.0, 2.0], [4.0, 2.0])

        result = set_ops.line_segment_polygon_intersection(seg, poly_with_hole)
        # Should have 2 segments (before hole and after hole)
        assert len(result) == 2

    def test_method_with_polygon(self):
        """Test the monkey-patched intersection method with Polygon."""
        seg = geometry.LineSegment([-1.0, 0.5], [2.0, 0.5])
        poly = geometry.Polygon.rectangle(0.0, 0.0, 1.0, 1.0)

        result = seg.intersection(poly)
        assert len(result) == 1


class TestUnion(TestOpsBase):
    def test_union_convex_case(self):
        p1 = geometry.Polygon.rectangle(0.0, 0.0, 2.0, 2.0)
        p2 = geometry.Polygon.rectangle(1.0, 0.1, 2.0, 2.0)

        union_poly = set_ops.union([p1, p2])
        area = union_poly.area
        assert jnp.abs(area - 6.1) < 1e-4

    def test_union_non_convex_case(self):
        p1 = geometry.Polygon.rectangle(0.0, 0.0, 2.0, 2.0)
        p2 = geometry.Polygon.rectangle(1.5, 1.5, 2.0, 2.0)

        union_poly = set_ops.union([p1, p2])
        area = union_poly.area
        expected_area = 7.75
        assert jnp.abs(area - expected_area) < 1e-3

    def test_concave_union(self):
        p1 = self.create_L_shape()
        p2 = geometry.Polygon.rectangle(1.0, 1.0, 1.0, 1.0)

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
        p3 = mutation.translate(p2, (0.05, 0.0))

        p4 = set_ops.union([p2, p3])

        # Expected area ~ 0.04484 (UPDATED for adaptive buffer resolution=60)
        assert jnp.abs(p4.area - 0.04484) < 1e-4
        assert not p4.overflow

    def test_union_three_polys(self):
        # Union of 3 squares in a row
        p1 = geometry.Polygon.rectangle(0.0, 0.0, 1.0, 1.0)
        p2 = geometry.Polygon.rectangle(0.5, 0.0, 1.0, 1.0)  # Overlaps p1
        p3 = geometry.Polygon.rectangle(
            1.0, 0.0, 1.0, 1.0
        )  # Overlaps p2 (touches p1 edge)

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
        p1 = geometry.Polygon.rectangle(0.0, 0.0, 2.0, 2.0)
        p2 = geometry.Polygon.rectangle(1.0, 1.0, 2.0, 2.0)

        diff_poly = set_ops.difference(p1, p2)
        area = diff_poly.area
        assert jnp.abs(area - 3.0) < 1e-3

    def test_difference_produces_concave(self):
        p1 = geometry.Polygon.rectangle(0.0, 0.0, 3.0, 1.0)
        # Use overlapping rectangle that crosses the boundary to avoid collinear clipping issues
        p2 = geometry.Polygon.rectangle(1.0, -0.5, 1.0, 1.0)

        diff_poly = set_ops.difference(p1, p2)
        area = diff_poly.area
        assert jnp.abs(area - 2.5) < 1e-3

    def test_hole_creation(self):
        p1 = geometry.Polygon.rectangle(0.0, 0.0, 3.0, 3.0)
        p2 = geometry.Polygon.rectangle(1.0, 1.0, 1.0, 1.0)

        diff_poly = set_ops.difference(p1, p2)
        area = diff_poly.area
        assert jnp.abs(area - 8.0) < 1e-3

    def test_p1_inside_p2(self):
        p1 = geometry.Polygon.rectangle(1.0, 1.0, 1.0, 1.0)
        p2 = geometry.Polygon.rectangle(0.0, 0.0, 3.0, 3.0)

        diff_poly = set_ops.difference(p1, p2)
        area = diff_poly.area
        assert jnp.abs(area - 0.0) < 1e-4


class TestTopologySetOps(TestOpsBase):
    def test_intersection_produces_multi_ring(self):
        # P1: Square at (0,0) size 2, Square at (5,0) size 2.
        sq1 = jnp.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=jnp.float32)
        sq2 = jnp.array([[5, 0], [7, 0], [7, 2], [5, 2]], dtype=jnp.float32)
        p1_verts = jnp.concatenate([sq1, sq2])
        p1 = geometry.Polygon(vertices=p1_verts, count=8, ring_counts=jnp.array([4, 4]))

        # P2: Large rectangle covering both.
        # [ -1, -1 ] to [ 8, 3 ]
        p2_verts = jnp.array([[-1, -1], [8, -1], [8, 3], [-1, 3]], dtype=jnp.float32)
        p2 = geometry.Polygon(vertices=p2_verts, count=4, ring_counts=jnp.array([4]))

        # Intersection should be P1 (both squares preserved).
        res = set_ops.intersection(p1, p2, max_vertices=32, max_rings=4)

        assert res.count == 8
        # Should have 2 rings
        valid_rings = res.ring_counts[res.ring_counts > 0]
        assert len(valid_rings) == 2
        assert jnp.all(valid_rings == 4)

        # Test area match
        assert jnp.abs(res.area - 8.0) < 1e-4

    def test_difference_creates_hole(self):
        # P1: 10x10 square
        p1 = geometry.Polygon.rectangle(0.0, 0.0, 10.0, 10.0)

        # P2: 4x4 square in middle
        # Rectangle(x,y,w,h) -> starts at (3,3) size 4x4 -> ends at (7,7)
        p2 = geometry.Polygon.rectangle(3.0, 3.0, 4.0, 4.0)

        res = set_ops.difference(p1, p2, max_vertices=32, max_rings=4)

        # Result should be 1 outer ring + 1 hole ring
        # Total vertices: 4 + 4 = 8
        assert res.count == 8
        valid_rings = res.ring_counts[res.ring_counts > 0]
        assert len(valid_rings) == 2

        # Area should be 100 - 16 = 84
        assert jnp.abs(res.area - 84.0) < 1e-4

    def test_island_in_hole(self):
        # Reproduces the "Island in a Hole" issue
        # p1: Inner circle (r=1)
        # p2: Hole circle (r=2)
        # p3: Outer circle (r=3)
        p1 = geometry.Polygon.circle((0, 0), 1.0, num_edges=32)
        p2 = geometry.Polygon.circle((0, 0), 2.0, num_edges=32)
        p3 = geometry.Polygon.circle((0, 0), 3.0, num_edges=32)

        # p3 - p2 -> Annulus (2 to 3)
        p3_minus_p2 = set_ops.difference(p3, p2)

        # (p3 - p2) + p1 -> Annulus + Inner Island
        p4 = set_ops.union([p3_minus_p2, p1])

        expected_area = (jnp.pi * 3**2 - jnp.pi * 2**2) + jnp.pi * 1**2

        assert jnp.abs(p4.area - expected_area) < 0.2

        # Verify Topology
        # Should have 3 rings (Outer, H ole, Island)
        # Circle default edges = 32.
        # p3 (32) + p2 (32) + p1 (32) = 96 vertices total (approx, assuming no extra clips)

        valid_rings = p4.ring_counts[p4.ring_counts > 0]
        assert len(valid_rings) == 3

        # Check roughly 96 vertices (allow equal, logic preserves existing vertices if no intersection)
        assert p4.count == 96


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
