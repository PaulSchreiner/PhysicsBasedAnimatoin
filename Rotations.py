"""
Helper functions for rotation math (all rotations are in a right handed coordinate frame)

"""

import numpy as np

def rotation_matrix_x(phi):
    c_t = np.cos(phi)
    s_t = np.sin(phi)


    return np.array([
        [1., 0., 0.],
        [0., c_t, -s_t],
        [0., s_t, c_t]
    ])


def rotation_matrix_y(phi):
    c_t = np.cos(phi)
    s_t = np.sin(phi)

    return np.array([
        [c_t, 0., s_t],
        [0., 1., 0.],
        [-s_t, 0., c_t]
    ])


def rotation_matrix_z(phi):
    c_t = np.cos(phi)
    s_t = np.sin(phi)

    return np.array([
        [c_t, -s_t, 0.],
        [s_t, c_t, 0.],
        [0., 0., 1.]
    ])


def rotation_matrix(phi, axis='x'):
    if axis == 'z':
        return rotation_matrix_z(phi)
    elif axis == 'y':
        return rotation_matrix_y(phi)
    elif axis == 'x':
        return rotation_matrix_x(phi)
    else:
        raise NotImplementedError("Axis type {} is not supported!".format(axis))


def compute_rotation_chain(theta, rotation_axis=None):
    """ Computes the rotation chain:
    .. math::
        T_i = T_0T_1 \dots T_N

    :param theta: array of revolute joint angles
    :param rotation_axis: optional, array of rotation axis. Can be 'x', 'y' or 'z'. Future implementations will also
    allow a list [x, y, z] for arbitrary joint axes.
    :return:
    :return:
    """
    if rotation_axis is None:
        # Set the rotation axis to x per default
        rotation_axis = ['x' for t in theta]

    R_in = np.eye(3)
    for phi, ax_ in zip(theta, rotation_axis):
        T = rotation_matrix(phi, ax_)
        R_in = forward_chain_rotations(R_in, T)
    return R_in


def forward_chain_rotations(R_in, T):
    """Caclulate the chain product:

        .. math::
            R_{in, i+1} = R_{in, i}T_i

    :param R_in:
    :param T:
    :return:
    """
    return np.matmul(R_in, T)


def backward_chain_rotations(R_in, T):
    """Caclulate the chain product:

        .. math::
            R_{in, i} = T_iR_{in, i+1}

    :param R_in:
    :param T:
    :return:
    """
    return np.matmul(T, R_in)
