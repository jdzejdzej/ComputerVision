import numpy as np
from scipy.linalg import rq


def get_projection_matrix(pts_3d, pts_2d):
    if pts_3d.shape[0] != pts_2d.shape[0]:
        raise ValueError('incompatible arrays')
    x_dim = 11
    y_dim = 2*pts_3d.shape[0]
    A = np.zeros((y_dim, x_dim))
    A[::2, :3] = pts_3d
    A[::2, 3] = 1
    A[1::2, 4:7] = pts_3d
    A[1::2, 7] = 1
    A[::2, -3:] = -pts_3d * pts_2d[:, 0, None]
    A[1::2, -3:] = -pts_3d * pts_2d[:, 1, None]
    x, res, r, s = np.linalg.lstsq(A, pts_2d.reshape((1, y_dim))[0])
    return np.append(x, 1).reshape(3, 4)


def compute_reprojection_error(pts_3d, pts_2d):
    M = get_projection_matrix(pts_3d, pts_2d)
    y_dim = pts_3d.shape[0]
    pts_2d_calculated = M[:, None, :].dot(np.hstack((pts_3d, np.ones((y_dim, 1)))).T)[:,0,:].T
    pts_2d_calculated = (pts_2d_calculated.T / pts_2d_calculated[:, 2]).T

    return np.linalg.norm(pts_2d - pts_2d_calculated[:, :2])


def factor(P):
    K, R = rq(P[:, :3])
    T = np.diag(np.sign(np.diag(K)))
    if np.linalg.det(T) < 0:
        T[1, 1] *= -1
    K = np.dot(K, T)
    R = np.dot(T, R)
    t = np.dot(np.linalg.inv(K), P[:, 3])
    c = -np.dot(R.T, t)
    return K, R, t, c

