"""
Helper functions for transformation math (all rotations are in a right handed coordinate frame)
"""
import numpy as np

from Transformations.Rotations import rotation_matrix


def compose_T(R, p):
    """Composes a homogeneous tranformation matrix from rotation matrix R and position 3vector p.

    :param R: Rotation matrix
    :param p: Position 3 vector
    :return:
    np.ndarray(4, 4)
    """
    T = np.zeros([4, 4])
    T[:3, :3] = R
    T[:3, 3] = p
    T[3, 3] = 1.

    return T


def compute_transformation_chain(theta, p, rotation_axes=None):
    """ Computes the transformation chain:
    .. math::
        T_i = T_0T_1 ... T_N

    :param theta: hierarchically ordered array of revolute joint angles
    :param: p: [mx3] array of positions representing the joint position with respect to its parent. Ordering must be
    the same as theta.
    :param rotation_axes: optional, array of rotation axis. Can be 'x', 'y' or 'z'. Future implementations will also
    allow an array [x, y, z] for arbitrary joint axes. Ordering must be the same as theta.
    :return:
    """
    if rotation_axes is None:
        # Set the rotation axis to x per default
        rotation_axes = ['x' for t in theta]

    T_in = np.eye(4)
    for phi, p_, ax_ in zip(theta, p, rotation_axes):
        R = rotation_matrix(phi, ax_)
        T = compose_T(R, p_)

        T_in = forward_chain_transformation(T_in, T)
    return T_in


def forward_chain_transformation(T_in, T):
    """Caclulate the chain product:

        .. math::
            T_{in, i+1} = T_{in, i}T_i

    :param T_in:
    :param T:
    :return:
    """
    return np.matmul(T_in, T)


def backward_chain_transformation(T_in, T):
    """Caclulate the chain product:

        .. math::
            T_{in, i} = T_iT_{in, i+1}

    :param T_in:
    :param T:
    :return:
    """
    return np.matmul(T, T_in)


def t_matrix_inverse(T):
    T_inv = np.zeros([4, 4])
    R_inv = T[:3, :3].T
    p = T[:3, 3]
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = -np.matmul(R_inv, p)
    T_inv[3, 3] = 1.

    return T_inv
