import jax
import jax.numpy as jnp
import pytest
from dshape import planarize


class TestPlanarize:
    def test_simple_intersection(self):
        # Two segments crossing forming an X
        # Seg 1: (0,0) -> (2,2)
        # Seg 2: (0,2) -> (2,0)
        # Intersection at (1,1)

        segments = jnp.array(
            [[[0.0, 0.0], [2.0, 2.0]], [[0.0, 2.0], [2.0, 0.0]]], dtype=jnp.float32
        )

        count = 2

        _, t_matrix, mask = planarize._find_all_intersections(segments, count)

        # Should have intersection at t=0.5 for both
        assert jnp.any(mask)

        # t_values where mask is true should be 0.5
        valid_ts = t_matrix[mask]
        assert jnp.allclose(valid_ts, 0.5)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
