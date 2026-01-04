import jax
import jax.numpy as jnp
import pytest
from dshape import core


class TestCore:
    def test_is_point_in_polygon(self):
        # Simple square
        vertices = jnp.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=jnp.float32)
        count = 4

        # Point inside
        p_in = jnp.array([1.0, 1.0])
        assert core._is_point_in_polygon(p_in, vertices, count)

        # Point outside
        p_out = jnp.array([3.0, 1.0])
        assert not core._is_point_in_polygon(p_out, vertices, count)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
