# dshape: Differentiable Shape Operations with JAX

`dshape` is a Python library for performing differentiable geometric operations on polygons, built on top of [JAX](https://github.com/google/jax). It enables gradient-based optimization of geometric shapes by providing differentiable implementations of common set and mutation operations.

## Features

- **Differentiable Set Operations** (`dshape.set_ops`):
    - `intersection(p1, p2)`: Sutherland-Hodgman clipping for polygons.
    - `union([p1, p2, ...])`: Construct unified geometry from a sequence of polygons.
    - `difference(p1, p2)`: Subtract one polygon from another.
- **Differentiable Mutations** (`dshape.mutation`):
    - `buffer(p, distance)`: Dilate (>0) or erode (<0) polygons with straight (miter) or rounded (arc) corners.
    - `offset(p, dx, dy)`: Translate polygons.
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
```

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
