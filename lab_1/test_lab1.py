import numpy as np
import pytest
from lab1 import get_projection_matrix, compute_reprojection_error

data_result = np.array([[-0.4583, 0.2947, 0.0139, -0.004],
                        [0.0509, 0.0546, 0.541, 0.0524],
                        [-0.109, -0.1784, 0.0443, -0.5968]])

pts_2d_norm = np.loadtxt('data/task12/pts2d-norm-pic_a.txt')
pts_3d_norm = np.loadtxt('data/task12/pts3d-norm.txt')

pts_2d = np.loadtxt('data/task12/pts2d-pic_a.txt')
pts_3d = np.loadtxt('data/task12/pts3d.txt')


def test_get_projection_matrix_input_arrays_dont_match_raises_value_error():
    a1 = np.array([1, 2])
    a2 = np.array([[1], [1]])
    with pytest.raises(ValueError) as err:
        get_projection_matrix(a1, a2)
        assert 'incompatible arrays' in str(err.value)


def test_get_projection_matrix_data_from_file():
    factor = -0.5968
    result = get_projection_matrix(pts_3d_norm, pts_2d_norm)
    np.testing.assert_array_almost_equal(result*factor, data_result, decimal=4)


def test_compute_reprojection_error_input_points_normalized():
    result = compute_reprojection_error(pts_3d_norm, pts_2d_norm)
    assert result == pytest.approx(0.012990, 10e-5)


def test_compute_reprojection_error_input_points_not_normalized():
    result = compute_reprojection_error(pts_3d, pts_2d)
    assert result == pytest.approx(3.966324, 10e-5)
