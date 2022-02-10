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


def jacobian_revolute_joint_axis_naive(phi, axis):
    if type(axis) == list:
        axis = np.array(axis)
    if axis.ndim > 1:
        if axis.shape[1] == 3:
            axis = axis[0]
        elif axis.shape[0] == 3:
            axis = axis[..., 0]
        else:
            raise ValueError("Axis must be an array of length 3, {} is not a valid format.".format(axis.shape))
    nrm = np.linalg.norm(axis)
    if nrm != 1:
        axis = axis / nrm

    ux = np.array([
        [0., -axis[2], axis[1]],
        [axis[2], 0., -axis[0]],
        [-axis[1], axis[0], 0.]
    ])

    R_ = rotation_matrix(phi, axis)

    J_ = np.zeros([4, 4])
    J_[:3, :3] = np.matmul(ux, R_)

    return J_


def jacobian_revolute_joint_naive(phi, axis='x'):
    if axis == 'z':
        return jacobian_revolute_joint_z_naive(phi)
    elif axis == 'y':
        return jacobian_revolute_joint_y_naive(phi)
    elif axis == 'x':
        return jacobian_revolute_joint_x_naive(phi)
    elif type(axis) == list:
        return jacobian_revolute_joint_axis_naive(phi, axis)
    else:
        raise NotImplementedError("Axis type {} is not supported!".format(axis))


def compute_joint_jacobian(phi, joint_type, **kwargs):
    """Compute the jacobian of a joint.

    :param phi: joint parameter(s)
    :param joint_type: str, type of the joint (revolute, prismatic, ball)
    :param kwargs: joint_type dependent arguments. See the specific joint type function.
    :return:
    """
    options_ = {
        "axis": 'x'
    }
    for k_, v_ in kwargs.items():
        options_[k_] = v_
    if joint_type == "revolute":
        axis = options_["axis"]
        if type(axis) == list:
            return jacobian_revolute_joint_axis_naive(phi, axis)
        elif axis == 'z':
            return jacobian_revolute_joint_z_naive(phi)
        elif axis == 'y':
            return jacobian_revolute_joint_y_naive(phi)
        elif axis == 'x':
            return jacobian_revolute_joint_x_naive(phi)
        else:
            raise NotImplementedError("Axis type {} is not supported!".format(axis))
    else:
        raise NotImplementedError("Joints of type {} are not implemented!".format(joint_type))


def compute_jacobian_column(A, dT_i, B, end_eff=None, i_=None, j_=None):
    """Computes the Jacobian of link i as:
    .. math::
        J_i = A*dT_i*B*e

    :param A: Chain of transforms up to link i
    :param dT_i: Partial derivative of the transform at link i
    :param B: Chain of transforms from link i + 1 until the end effector
    :param end_eff: optional, end effector position. Default = [0. 0. 1.]
    :param i_: TODO
    :param j_: TODO
    :return:
    """
    if end_eff is None:
        end_eff = np.array([0., 0., 0., 1.])

    temp0 = np.matmul(A, dT_i)
    temp0 = np.matmul(temp0, B)
    temp1 = np.matmul(temp0, end_eff)
    if i_ is None:
        return temp1[:3]
    else:
        temp2 = np.matmul(temp0, i_)
        temp3 = np.matmul(temp0, j_)
        return np.concatenate([temp1[:3], temp2[:3], temp3[:3]])


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
        end_eff = np.array([0., 0., 0., 1.])

    assert len(rotation_axes) == theta.shape[0], \
        "Sizes of rotation_axes ({}) and theta ({}) do not correspond!".format(len(rotation_axes), theta.shape[0])
    assert p.shape[0] == theta.shape[0], \
        "Sizes of p ({}) and theta ({}) do not correspond!".format(p.shape[0], theta.shape[0])


    J = np.zeros([3, len(theta)])
    A = np.eye(4)
    B = compute_transformation_chain(theta, p, rotation_axes)
    T_i = compose_T(rotation_matrix(theta[0], rotation_axes[0]), p[0])

    for ii, (phi, p_, ax_) in enumerate(zip(theta, p, rotation_axes)):
        B = backward_chain_transformation(B, t_matrix_inverse(T_i))
        J_ii = jacobian_revolute_joint_naive(phi, ax_)
        J[:, ii] = compute_jacobian_column(A, J_ii, B, end_eff)[:3]
        T_i = compose_T(rotation_matrix(phi, ax_), p_)
        A = forward_chain_transformation(A, T_i)


    return J, np.matmul(A, end_eff)


def compute_jacobian(theta, p, joint_axes=None, end_eff=None, i_=None, j_=None):
    """Computes the Jacobian for a chain of joints

    :param theta: array of revolute joint angles
    :param: p: [mx3] array of positions representing the joint position with respect to its parent. Ordering must be
    the same as theta.
    :param joint_axes: optional, array of rotation axis. Can be 'x', 'y' or 'z'. Future implementations will also
    allow a list [x, y, z] for arbitrary joint axes.
    :param end_eff: optional, array containing the offset of the end effector from the frame origin
    :param i_: TODO
    :param j_: TODO
    :return:
    np.ndarray containing the jacobian of the chain
    """
    if joint_axes is None:
        # Set the rotation axis to x per default
        joint_axes = ['x' for t in theta]

    if end_eff is None:
        end_eff = np.array([0., 0., 0., 1.])

    if i_ is None or j_ is None:
        J = np.zeros([3, len(theta)])
    else:
        J = np.zeros([9, len(theta)])

    assert len(joint_axes) == theta.shape[0], \
        "Sizes of rotation_axes ({}) and theta ({}) do not correspond!".format(len(joint_axes), theta.shape[0])
    assert p.shape[0] == theta.shape[0], \
        "Sizes of p ({}) and theta ({}) do not correspond!".format(p.shape[0], theta.shape[0])

    A = np.eye(4)
    B = compute_transformation_chain(theta, p, joint_axes)
    T_i = compose_T(rotation_matrix(theta[0], joint_axes[0]), p[0])

    for ii, (phi, p_, ax_) in enumerate(zip(theta, p, joint_axes)):
        B = backward_chain_transformation(B, t_matrix_inverse(T_i))
        J_ii = compute_joint_jacobian(phi, joint_type='revolute', axis=ax_)
        J[:, ii] = compute_jacobian_column(A, J_ii, B, end_eff, i_, j_)
        T_i = compose_T(rotation_matrix(phi, ax_), p_)
        A = forward_chain_transformation(A, T_i)

    if i_ is None:
        return J, np.matmul(A, end_eff)[:3]
    else:
        return J, np.concatenate([np.matmul(A, end_eff)[:3], np.matmul(A, j_)[:3], np.matmul(A, j_)[:3]])

def right_sided_pseudo_inverse(J, lambda_):
    rnk = np.linalg.matrix_rank(J)
    JJT_inv = np.zeros([J.shape[0], J.shape[0]])
    if lambda_:
        JJT_inv[:rnk, :rnk] = np.linalg.inv(np.matmul(J[:rnk, :], J[:rnk, :].T) + np.eye(rnk) * lambda_)
    else:
        JJT_inv[:rnk, :rnk] = np.linalg.inv(np.matmul(J[:rnk, :], J[:rnk, :].T))
    return np.matmul(J.T, JJT_inv)


def left_sided_pseudo_inverse(J, lambda_):
    if lambda_:
        JTJ_inv = np.linalg.inv(np.matmul(J.T, J) + np.eye(J.shape[1]) * lambda_)
    else:
        JTJ_inv = np.linalg.inv(np.matmul(J.T, J))
    return np.matmul(JTJ_inv, J.T)


def compute_jacobian_inverse(J, lambda_=0, use_np=False):
    """Compute the More-Penrose pseudo inverse. (This can probably be optimised by using numpy's pinv function, but for
    the purpose of exersice I implemented it myself)

    :param J: Jacobian
    :param lambda_: optional, regularisation parameter.
    :return:
    np.ndarray, inverse or pseudo inverse of the Jacobian
    """

    if J.shape[0] == J.shape[1] and not lambda_:
        return np.linalg.inv(J)
    elif use_np:
        return np.linalg.pinv(J)
    elif J.shape[0] > J.shape[1]:
        return left_sided_pseudo_inverse(J, lambda_)
    else:
        return right_sided_pseudo_inverse(J, lambda_)


