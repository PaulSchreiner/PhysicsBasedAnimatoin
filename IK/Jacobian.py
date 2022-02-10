"""
This file contains computational methods for obtaining the Jacobian for different joint types

"""

import numpy as np

# from Rotations import rotation_matrix, forward_chain_rotations, backward_chain_rotations, compute_rotation_chain
from Transformations.Rotations import rotation_matrix
from Transformations.Transformations import forward_chain_transformation, backward_chain_transformation, compute_transformation_chain, \
    compose_T, t_matrix_inverse


def jacobian_revolute_joint_x_naive(phi):
    c_t = np.cos(phi)
    s_t = np.sin(phi)

    return np.array([
        [0., 0., 0., 0.],
        [0., -s_t, -c_t, 0.],
        [0., c_t, -s_t, 0.],
        [0., 0., 0., 0.]
    ])


def jacobian_revolute_joint_y_naive(phi):
    c_t = np.cos(phi)
    s_t = np.sin(phi)

    return np.array([
        [-s_t, 0., c_t, 0.],
        [0., 0., 0., 0.],
        [-c_t, 0., -s_t, 0.],
        [0., 0., 0., 0.]
    ])


def jacobian_revolute_joint_z_naive(phi):
    c_t = np.cos(phi)
    s_t = np.sin(phi)

    return np.array([
        [-s_t, -c_t, 0., 0.],
        [c_t, -s_t, 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]
    ])


def jacobian_revolute_joint_naive(phi, axis='x'):
    if axis == 'z':
        return jacobian_revolute_joint_z_naive(phi)
    elif axis == 'y':
        return jacobian_revolute_joint_y_naive(phi)
    elif axis == 'x':
        return jacobian_revolute_joint_x_naive(phi)
    else:
        raise NotImplementedError("Axis type {} is not supported!".format(axis))


def compute_revolute_jacobian(A, dT_i, B, end_eff=None):
    """Computes the Jacobian of link i as:
    .. math::
        J_i = A*dT_i*B*e

    :param A: Chain of transforms up to link i
    :param dT_i: Partial derivative of the transform at link i
    :param B: Chain of transforms from link i + 1 until the end effector
    :param end_eff: optional, end effector position. Default = [0. 0. 1.]
    :return:
    """
    if end_eff is None:
        end_eff = np.array([0., 0., 1., 1.])

    temp = np.matmul(A, dT_i)
    temp = np.matmul(temp, B)
    temp = np.matmul(temp, end_eff)
    return temp


def compute_jacobian_revolute_chain(theta, p, rotation_axes=None, end_eff=None):
    """Computes the Jacobian for a chain of revolute joints

    :param theta: array of revolute joint angles
    :param: p: [mx3] array of positions representing the joint position with respect to its parent. Ordering must be
    the same as theta.
    :param rotation_axes: optional, array of rotation axis. Can be 'x', 'y' or 'z'. Future implementations will also
    allow a list [x, y, z] for arbitrary joint axes.
    :param end_eff: optional, array containing the offset of the end effector from the frame origin
    :return:
    np.ndarray containing the jacobian of the chain
    """
    if rotation_axes is None:
        # Set the rotation axis to x per default
        rotation_axes = ['x' for t in theta]

    if end_eff is None:
        end_eff = np.array([0., 0., 1., 1.])

    J = np.zeros([4, len(theta)])
    A = np.eye(4)
    B = compute_transformation_chain(theta[1:], p[1:], rotation_axes)

    for ii, (phi, p_, ax_) in enumerate(zip(theta, p, rotation_axes)):
        J_ii = jacobian_revolute_joint_naive(phi, ax_)
        J[:, ii] = compute_revolute_jacobian(A, J_ii, B, end_eff)
        T_i = compose_T(rotation_matrix(phi, ax_), p_)
        A = forward_chain_transformation(A, T_i)
        B = backward_chain_transformation(B, t_matrix_inverse(T_i))

    return J, np.matmul(A, end_eff)


def right_sided_pseudo_inverse(J):
    JJT_inv = np.linalg.inv(np.matmul(J, J.T))
    return np.matmul(J.T, JJT_inv)


def left_sided_pseudo_inverse(J):
    JTJ_inv = np.linalg.inv(np.matmul(J.T, J))
    return np.matmul(JTJ_inv, J.T)

def compute_jacobian_inverse(J):
    """Compute the More-Penrose pseudo inverse. (This can probably be optimised by using numpy's pinv function, but for
    the purpose of exersice I implemented it myself)

    :param J: Jacobian
    :return:
    np.ndarray, inverse or pseudo inverse of the Jacobian
    """
    if J.shape[0] == J.shape[1]:
        return np.linalg.inv(J)
    elif J.shape[0] > J.shape[1]:
        return left_sided_pseudo_inverse(J)
    else:
        return right_sided_pseudo_inverse(J)


