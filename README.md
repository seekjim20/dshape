# dshape: Differentiable Shape Operations with JAX

`dshape` is a Python library for performing differentiable geometric operations on polygons, built on top of [JAX](https://github.com/google/jax). It enables gradient-based optimization of geometric shapes by providing differentiable implementations of common set and mutation operations.

## Features

- **Differentiable Mutations** (`dshape.mutation`):
    - `offset(p, dxy)`: Translate polygons.
    - `rotate(p, angle, center)`: Rotate polygons around a point.
    - `scale(p, factor, origin)`: Scale polygons (uniform/non-uniform) from an origin.
    - `buffer(p, distance)`: Dilate (>0) or erode (<0) polygons with straight (miter) or rounded (arc) corners.
- **Differentiable Set Operations** (`dshape.set_ops`):
    - `intersection(p1, p2)`: Sutherland-Hodgman clipping for polygons.
    - `union([p1, p2, ...])`: Construct unified geometry from a sequence of polygons.
    - `difference(p1, p2)`: Subtract one polygon from another.
- **Constructive Geometry** (`dshape.constructive`):
    - `convex_hull(p)`: Computes the convex hull of a polygon (including multi-ring inputs).
- **JAX Integration**: Fully compatible with `jax.jit`, `jax.grad`, and `jax.vmap`.
- **Robustness**: Handles degenerate cases, self-intersections (pruning), and floating-point edge cases.

## Usage

### Functional API

The primary API uses module-level functions that operate on `Polygon` objects.

```python
import jax.numpy as jnp
from dshape import geometry, set_ops, mutation

# Define a polygon
verts = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
poly = geometry.Polygon(vertices=verts, count=3)

# Offset (Translation)
offset = mutation.offset(poly, dxy=(0.1, 0.1))

# Rotate (Rotation)
rotated = mutation.rotate(poly, angle_rad=jnp.pi/4, center=[0.5, 0.5])

# Scale (Scaling)
scaled = mutation.scale(poly, factor=2.0, origin=[0, 0])

# Buffer (Dilation)
buffered = mutation.buffer(poly, distance=0.1)

# Buffer (Erosion)
eroded = mutation.buffer(poly, distance=-0.1)

# Set Operations
p1 = geometry.Rectangle(0.0, 0.0, 1.0, 1.0)
p2 = geometry.Rectangle(0.5, 0.5, 1.0, 1.0)

union_res = set_ops.union([p1, p2])
inter_res = set_ops.intersection(p1, p2)
diff_res  = set_ops.difference(p1, p2)

# Convex Hull
hull = constructive.convex_hull(p1)
```

### Object-Oriented API (Methods & Operators)

For convenience, `dshape` provides method shortcuts and operator overloads directly on the `Polygon` class.

```python
# Mutation Methods
p_buf = p1.buffer(0.1)
p_off = p1.offset((1.0, 1.0))
p_rot = p1.rotate(jnp.pi/2, center=[0, 0])
p_scl = p1.scale(2.0, origin=[0, 0])

# Set Operations (Operators)
p_union = p1 + p2  # Union
p_diff  = p1 - p2  # Difference
p_inter = p1 * p2  # Intersection

# properties
hull = p1.convex_hull
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

- `src/dshape/`: Source code.
    - `geometry.py`: Core `Polygon` class.
    - `set_ops.py`: Intersection, Union, Difference.
    - `mutation.py`: Buffer, Offset, Rotate, Scale.
    - `constructive.py`: Convex Hull.
    - `planarize.py`: Robust planarization logic (graph reconstruction).
    - `plotting.py`: Visualization helpers (Plotly).
    - `core.py`: Internal geometric helpers (JAX kernels).
- `tests/`: Unit tests (using `pytest`).

## Testing

Run the test suite with:

```bash
pytest tests/
```
