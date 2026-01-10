import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dshape import geometry


class TestLineSegment:
    def test_line_segment_creation(self):
        seg = geometry.LineSegment([0.0, 0.0], [3.0, 4.0])
        assert seg.p1.shape == (2,)
        assert seg.p2.shape == (2,)
        assert jnp.allclose(seg.p1, jnp.array([0.0, 0.0]))
        assert jnp.allclose(seg.p2, jnp.array([3.0, 4.0]))

    def test_line_segment_length(self):
        # 3-4-5 triangle
        seg = geometry.LineSegment([0.0, 0.0], [3.0, 4.0])
        assert jnp.abs(seg.length - 5.0) < 1e-6

    def test_line_segment_midpoint(self):
        seg = geometry.LineSegment([0.0, 0.0], [4.0, 6.0])
        expected = jnp.array([2.0, 3.0])
        assert jnp.allclose(seg.midpoint, expected)

    def test_line_segment_direction(self):
        seg = geometry.LineSegment([0.0, 0.0], [3.0, 4.0])
        # Unit vector: (3/5, 4/5)
        expected = jnp.array([0.6, 0.8])
        assert jnp.allclose(seg.direction, expected, atol=1e-6)

    def test_line_segment_from_arrays(self):
        p1 = jnp.array([1.0, 2.0])
        p2 = jnp.array([4.0, 6.0])
        seg = geometry.LineSegment(p1, p2)
        assert jnp.abs(seg.length - 5.0) < 1e-6

    def test_line_segment_jit(self):
        seg = geometry.LineSegment([0.0, 0.0], [3.0, 4.0])

        @jax.jit
        def length_fn(s):
            return s.length

        res = length_fn(seg)
        assert jnp.abs(res - 5.0) < 1e-6

    def test_line_segment_in_trace(self):
        @jax.jit
        def f(s):
            return s.midpoint

        seg = geometry.LineSegment([0.0, 0.0], [4.0, 4.0])
        res = f(seg)
        assert jnp.allclose(res, jnp.array([2.0, 2.0]))

        assert jnp.allclose(res, jnp.array([2.0, 2.0]))


class TestPolygonContains:
    """Tests for Polygon.contains(point) functionality."""

    def test_simple_containment(self):
        """Test point containment in a simple polygon."""
        poly = geometry.Rectangle(0.0, 0.0, 2.0, 2.0)

        # Test point inside
        p_in = geometry.Point([1.0, 1.0])
        assert poly.contains(p_in)

        # Test point outside
        p_out = geometry.Point([3.0, 3.0])
        assert not poly.contains(p_out)

    def test_containment_array_input(self):
        """Test containment with raw array input."""
        poly = geometry.Rectangle(0.0, 0.0, 2.0, 2.0)
        assert poly.contains([1.0, 1.0])
        assert not poly.contains([3.0, 3.0])

    def test_hole_containment(self):
        """Test point containment with a hole."""
        # Outer: 4x4 square at (0,0) -> (-2,-2) to (2,2) ? No, Rectangle(x,y,w,h)
        # Rect(0,0,4,4) -> (0,0) to (4,4)
        outer = geometry.Rectangle(0.0, 0.0, 4.0, 4.0)
        # Hole: 2x2 square at (1,1) -> (1,1) to (3,3)
        hole = geometry.Rectangle(1.0, 1.0, 2.0, 2.0)

        poly = geometry.Polygon.from_exteriors_interiors([outer], [hole])

        # Test point in the "meat" of the polygon
        p_meat = geometry.Point([0.5, 0.5])
        assert poly.contains(p_meat)

        # Test point in the hole (should be False)
        p_hole = geometry.Point([2.0, 2.0])
        assert not poly.contains(p_hole)

        # Test point outside outer boundary
        p_out = geometry.Point([5.0, 5.0])
        assert not poly.contains(p_out)

    def test_island_containment(self):
        """Test point in an island inside a hole (Even-Odd rule)."""
        # Outer: Circle r=3
        p3 = geometry.Circle((0, 0), 3.0)
        # Hole: Circle r=2
        p2 = geometry.Circle((0, 0), 2.0)
        # Island: Circle r=1 (This is a union of (p3-p2) and p1)
        # But from_exteriors_interiors takes list of exteriors and interiors.
        # If we just pass multiple rings, how are they interpreted?
        # Even-Odd rule handles nested rings automatically regardless of explicit "hole" labeling,
        # providing they are properly oriented or just structurally nested.
        # Let's use set_ops to robustly construct an island polygon.

        try:
            from dshape import set_ops
        except ImportError:
            import sys

            # Skip if set_ops not available (though it should be)
            pass

        p1 = geometry.Circle((0, 0), 1.0)
        # Annulus p3 - p2
        annulus = set_ops.difference(p3, p2)
        # Island p1 + annulus
        poly_island = set_ops.union([annulus, p1])

        # Check point in island (r < 1) -> True
        assert poly_island.contains([0.0, 0.0])

        # Check point in "moat" (1 < r < 2) -> False
        # r=1.5
        assert not poly_island.contains([1.5, 0.0])

        # Check point in outer land (2 < r < 3) -> True
        # r=2.5
        assert poly_island.contains([2.5, 0.0])

        # Check point completely outside (r > 3) -> False
        assert not poly_island.contains([4.0, 0.0])


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

    def test_default_count(self):
        # Test that count defaults to shape[0] correctly
        vertices = jnp.array([[0, 0], [1, 0], [0, 1]], dtype=jnp.float32)
        poly = geometry.Polygon(vertices=vertices)  # No count provided

        assert poly.count == 3
        assert poly.area > 0


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
        circle = geometry.Circle((x, y), r, num_edges)

        assert circle.count == num_edges
        assert circle.vertices.shape[0] == num_edges

        expected_area = 0.5 * num_edges * r**2 * np.sin(2 * np.pi / num_edges)
        area = circle.area
        assert jnp.abs(area - expected_area) < 1e-4

    def test_circle_jit(self):
        x, y, r = 0.0, 0.0, 1.0
        circle = geometry.Circle((x, y), r, num_edges=32)

        @jax.jit
        def area_fn(p):
            return p.area

        res = area_fn(circle)
        assert res > 3.0

    def test_circle_in_trace(self):
        @jax.jit
        def f(c):
            return c.area

        circle = geometry.Circle((0, 0), 1)
        res = f(circle)
        assert res > 0


class TestPolygonAPI:
    def test_exteriors_interiors(self):
        # Create a square (CCW)
        sq_verts = jnp.array([[0, 0], [3, 0], [3, 3], [0, 3]], dtype=jnp.float32)
        p1 = geometry.Polygon(vertices=sq_verts, count=4)

        # Create a hole (CW)
        hole_verts = jnp.array([[1, 1], [1, 2], [2, 2], [2, 1]], dtype=jnp.float32)
        p2 = geometry.Polygon(vertices=hole_verts, count=4)

        # Combine manually
        combined_verts = jnp.concatenate([sq_verts, hole_verts])
        combined_rc = jnp.array([4, 4])
        poly = geometry.Polygon(
            vertices=combined_verts, count=8, ring_counts=combined_rc
        )

        # Check exteriors
        exts = poly.exteriors
        assert len(exts) == 1
        assert jnp.allclose(exts[0].vertices[:4], sq_verts)

        # Check interiors
        ints = poly.interiors
        assert len(ints) == 1
        assert jnp.allclose(ints[0].vertices[:4], hole_verts)

    def test_from_exteriors_interiors(self):
        # Create components
        ext = geometry.Rectangle(0, 0, 3, 3)
        # Create interior as a normal rectangle (positive area)
        # Factory should flip it to negative
        hole = geometry.Rectangle(1, 1, 1, 1)

        poly = geometry.Polygon.from_exteriors_interiors([ext], [hole])

        assert poly.count == 8
        assert len(poly.ring_counts) >= 2  # Might be padded
        assert poly.ring_counts[0] == 4
        assert poly.ring_counts[1] == 4

        # Check area: 9 - 1 = 8
        assert jnp.abs(poly.area - 8.0) < 1e-5

        # Check properties
        assert len(poly.exteriors) == 1
        assert len(poly.interiors) == 1

    def test_multi_exterior(self):
        r1 = geometry.Rectangle(0, 0, 1, 1)
        r2 = geometry.Rectangle(2, 2, 1, 1)

        poly = geometry.Polygon.from_exteriors_interiors([r1, r2])

        assert poly.count == 8
        assert poly.area == 2.0
        assert len(poly.exteriors) == 2
        assert len(poly.interiors) == 0


class TestTopologyGeometry:
    def test_multi_ring_area(self):
        # Create a square with a hole
        outer_verts = jnp.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=jnp.float32)
        hole_cw = jnp.array([[3, 3], [3, 7], [7, 7], [7, 3]], dtype=jnp.float32)  # CW

        # Pack
        all_verts = jnp.concatenate([outer_verts, hole_cw], axis=0)
        ring_counts = jnp.array([4, 4])

        poly = geometry.Polygon(vertices=all_verts, count=8, ring_counts=ring_counts)

        area = poly.area
        expected = 100.0 - 16.0  # 10*10 - 4*4
        assert jnp.abs(area - expected) < 1e-4

    def test_self_intersection_multi_ring(self):
        # Ring 1: Valid square
        # Ring 2: Bowtie (self-intersecting)
        sq = jnp.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=jnp.float32)
        bowtie = jnp.array([[5, 0], [7, 2], [5, 2], [7, 0]], dtype=jnp.float32)

        verts = jnp.concatenate([sq, bowtie])
        p = geometry.Polygon(vertices=verts, count=8, ring_counts=jnp.array([4, 4]))

        has_int = p.self_intersect
        assert has_int  # Because bowtie intersects itself

        # Case 2: Intersection BETWEEN rings
        sq1 = jnp.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=jnp.float32)
        sq2 = jnp.array([[3, 3], [7, 3], [7, 7], [3, 7]], dtype=jnp.float32)
        verts2 = jnp.concatenate([sq1, sq2])
        p2 = geometry.Polygon(vertices=verts2, count=8, ring_counts=jnp.array([4, 4]))
        assert p2.self_intersect


class TestOperators:
    def test_add_substitution(self):
        # p1 + p2 should be union
        p1 = geometry.Rectangle(0, 0, 1, 1)  # Area 1
        p2 = geometry.Rectangle(1, 0, 1, 1)  # Area 1, touches p1

        # Union area should be 2
        p3 = p1 + p2
        assert jnp.abs(p3.area - 2.0) < 1e-5

        # Check chaining
        p4 = geometry.Rectangle(2, 0, 1, 1)
        p5 = p1 + p2 + p4
        assert jnp.abs(p5.area - 3.0) < 1e-5

    def test_sub_substitution(self):
        # p1 - p2 should be difference
        p1 = geometry.Rectangle(0, 0, 2, 2)  # Area 4
        p2 = geometry.Rectangle(0.5, 0.5, 1, 1)  # Area 1, fully inside

        p3 = p1 - p2
        assert jnp.abs(p3.area - 3.0) < 1e-5

    def test_mul_substitution(self):
        # p1 * p2 should be intersection
        p1 = geometry.Rectangle(0, 0, 2, 2)  # Area 4
        p2 = geometry.Rectangle(1, 0, 2, 2)  # Intersection [1,0] to [2,2] -> Area 2

        p3 = p1 * p2
        assert jnp.abs(p3.area - 2.0) < 1e-5

    def test_convex_hull_property(self):
        # Verify p.convex_hull calls constructive
        p1 = geometry.Rectangle(0, 0, 1, 1)
        hull = p1.convex_hull

        # Valid property access
        assert isinstance(hull, geometry.Polygon)
        assert jnp.abs(hull.area - 1.0) < 1e-5


class TestMutationShortcuts:
    def test_translate_shortcut(self):
        p = geometry.Rectangle(0, 0, 1, 1)  # Area 1, Center (0.5, 0.5)
        p_moved = p.translate((1.0, 1.0))  # Center (1.5, 1.5)

        assert jnp.abs(p_moved.area - 1.0) < 1e-5
        # Check first vertex (0,0) -> (1,1)
        assert jnp.linalg.norm(p_moved.vertices[0] - jnp.array([1.0, 1.0])) < 1e-5

    def test_rotate_shortcut(self):
        p = geometry.Rectangle(0, 0, 2, 2)  # Center (1,1)
        p_rot = p.rotate(jnp.pi / 2, center=[1.0, 1.0])

        assert jnp.abs(p_rot.area - 4.0) < 1e-5
        assert p_rot.count == 4

    def test_scale_shortcut(self):
        p = geometry.Rectangle(0, 0, 1, 1)
        p_scaled = p.scale(2.0, origin=[0, 0])

        assert jnp.abs(p_scaled.area - 4.0) < 1e-5

    def test_buffer_shortcut(self):
        p = geometry.Rectangle(0, 0, 1, 1)
        p_buf = p.buffer(0.1)

        assert p_buf.area > 1.0
        assert p_buf.count > 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
