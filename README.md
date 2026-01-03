# dshape: Differentiable Shape Operations with JAX

`dshape` is a Python library for performing differentiable geometric operations on polygons, built on top of [JAX](https://github.com/google/jax). It enables gradient-based optimization of geometric shapes by providing differentiable implementations of common set and mutation operations.

## Features

- **Differentiable Set Operations** (`dshape.set_ops`):
    - `intersection(p1, p2)`: Sutherland-Hodgman clipping for polygons.
    - `union([p1, p2, ...])`: Construct unified geometry from a sequence of polygons.
    - `difference(p1, p2)`: Subtract one polygon from another.
- **Differentiable Mutations** (`dshape.mutation`):
    - `offset(p, dx, dy)`: Translate polygons.
    - `rotate(p, angle, center)`: Rotate polygons around a point.
    - `scale(p, factor, origin)`: Scale polygons (uniform/non-uniform) from an origin.
    - `buffer(p, distance)`: Dilate (>0) or erode (<0) polygons with straight (miter) or rounded (arc) corners.
- **Constructive Geometry** (`dshape.constructive`):
    - `convex_hull(p)`: Computes the convex hull of a polygon (including multi-ring inputs).
- **JAX Integration**: Fully compatible with `jax.jit`, `jax.grad`, and `jax.vmap`.
- **Robustness**: Handles degenerate cases, self-intersections (pruning), and floating-point edge cases.

## Usage

```python
import jax.numpy as jnp
from dshape import geometry, set_ops, mutation

# Define a polygon
verts = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
poly = geometry.Polygon(vertices=verts, count=3)

# Buffer it (Dilation)
buffered_poly = mutation.buffer(poly, distance=0.1)

# Intersect with another polygon
clip_poly = geometry.Rectangle(0.5, 0.5, 1.0, 1.0)
result = set_ops.intersection(buffered_poly, clip_poly)

# Union of multiple polygons
p1 = geometry.Rectangle(0.0, 0.0, 1.0, 1.0)
p2 = geometry.Rectangle(0.5, 0.0, 1.0, 1.0)
p3 = geometry.Rectangle(1.0, 0.0, 1.0, 1.0)
union_result = set_ops.union([p1, p2, p3])

# Rotate and Scale
rotated = mutation.rotate(p1, angle_rad=jnp.pi/4, center=[0.5, 0.5])
scaled = mutation.scale(p1, factor=2.0, origin=[0, 0])

# Operator Overloading (Syntactic Sugar)
# (+) Union
union_poly = p1 + p2
# (-) Difference
diff_poly = p1 - p2
# (*) Intersection
inter_poly = p1 * p2

# Convex Hull
from dshape import constructive
hull = constructive.convex_hull(p1)
# Or via property
hull_prop = p1.convex_hull
```

## Multi-Ring Polygons (Holes & Islands)

`dshape` supports polygons with multiple rings (e.g., shapes with holes, disjoint islands, or nested shells).

### Creating Polygons with Holes

You can construct multi-ring polygons using the `from_exteriors_interiors` factory method.
- **Exteriors**: List of polygons representing the outer boundaries (CCW winding, positive area).
- **Interiors**: List of polygons representing holes (CW winding, negative area). The factory automatically corrects winding if needed.

```python
# Construct a square with a triangular hole
outer_square = geometry.Rectangle(0.0, 0.0, 10.0, 10.0)
hole = geometry.Polygon(
    vertices=jnp.array([[2., 2.], [8., 2.], [5., 8.]]), 
    count=3
)

# The resulting polygon contains both rings
complex_shape = geometry.Polygon.from_exteriors_interiors(
    exteriors=[outer_square], 
    interiors=[hole]
)

# Properties
print(complex_shape.area)  # Square Area - Triangle Area
print(len(complex_shape.exteriors)) # 1
print(len(complex_shape.interiors)) # 1
```

### Disjoint Islands (Multi-Polygons)

A single `Polygon` object can represent multiple disjoint shapes by providing a list of exterior rings.

```python
# Create a shape consisting of two separate rectangles
rect1 = geometry.Rectangle(0, 0, 2, 2)
rect2 = geometry.Rectangle(5, 5, 2, 2)

multi_poly = geometry.Polygon.from_exteriors_interiors(
    exteriors=[rect1, rect2]
)
```

## Known Limitations

- **Fixed Buffer Sizes**: All JAX operations require static buffer sizes (e.g., `max_vertices`, `max_rings`).
    - If an operation (e.g., union of many shapes) produces more vertices/rings than the buffer allows, the `overflow` flag will be set on the result.
    - **Mitigation**: Check `.overflow` and retry with larger buffers if necessary, or conservatively size buffers for your domain.
- **Coincident Edge Clipping**: The current `set_ops.intersection` (Sutherland-Hodgman) implementation has known precision limitations when clipping edges that are perfectly overlapping or collinear. 
    - **Mitigation**: Robust handling is prioritized for general cases; perturb inputs slightly if degenerate overlaps cause dropping of edges.
- **Planarization Cost**: The robust erosion algorithm uses an O(N²) segment intersection check. While optimized with `vmap`, it may be slow for polygons with thousands of vertices.

## Structure

- `src/`: Source code.
    - `geometry.py`: Core `Polygon` class.
    - `set_ops.py`: Intersection, Union, Difference.
    - `mutation.py`: Buffer, Offset.
    - `core.py`: Internal geometric helpers.
- `test/`: Unit tests (using `pytest`).

## Testing

Run the test suite with:

```bash
pytest test/
```
