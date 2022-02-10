"""
This file contains an inverse kinematics solver library
"""
import numpy as np

from IK.Jacobian import compute_jacobian_revolute_chain, compute_jacobian_inverse, compute_jacobian


def base_ik_solver(func, target, p0, theta0=None, end_eff=None, i_=None, j_=None, tol=1.0e-3, max_iter=10,
                   rotation_axes=None, lr=1., print_freq=10):
    """ This is a base IK solver. It uses an update function func which returns the change in joint parameters, dtheta.

    :param func: update function that returns a change in joint parameter, dtheta
    :param target: target for the optimisation problem
    :param p0: joint offsets
    :param theta0: initial joint angles
    :param end_eff: end effector position with respect to its frame origin
    :param i_: TODO
    :param j_: TODO
    :param tol: tolerance, when the error drops bellow this number, the optimisation is considered converged.
    :param max_iter: max number of iterations before the optimisation is stopped
    :param rotation_axes: list(str) Rotation axes of the revolute joints, currently only 'x', 'y' and 'z' are accepted.
    Length must be equal to the length of p0. If None then all axes will be set to 'x'
    :param lr: Learning rate
    :param print_freq: optional, int, the error will be printed every print_freq iterations. Default = 10.
    :return:
    np.ndarray, computed joint angles.

    TODO: project angles to -pi <-> pi interval
    """
    if theta0 is None:
        theta0 = [0 for _ in p0]

    if end_eff is None:
        end_eff = np.array([0., 0., 0., 1.])

    theta = theta0

    for ii in range(max_iter):
        J, e = compute_jacobian(theta, p0, end_eff=end_eff, joint_axes=rotation_axes, i_=i_, j_=j_)
        err = np.linalg.norm(target - e)
        if not ii % print_freq:
            print("Error: {}".format(err))
        if tol > err:
            print("Converged after {} iterations! Breaking loop.\n".format(ii))
            return theta
        dtheta = func(J_=J, e_=e, target_=target, lr_=lr)
        theta += dtheta

    print("Max iterations reached, did not converge!\n")
    return theta


def pseudo_inverse_solver(target, p0, theta0=None, end_eff=None, i_=None, j_=None, tol=1.0e-4, max_iter=10,
                          rotation_axes=None, lr=1., print_freq=10, robust=True):
    """ This is an IK solver using the pseudo inverse of the Jacobian to solve the optimisation problem. In case of
    singularities (fx. gimbal lock), it will switch to using the Jacobian transpose instead. That last step is a bit
    hacky, but it kinda works to avoid gimbal locks.

    :param target: target for the optimisation problem
    :param p0: joint offsets
    :param theta0: initial joint angles
    :param end_eff: end effector position with respect to its frame origin
    :param i_: TODO
    :param j_: TODO
    :param tol: tolerance, when the error drops bellow this number, the optimisation is considered converged.
    :param max_iter: max number of iterations before the optimisation is stopped
    :param rotation_axes: list(str) Rotation axes of the revolute joints, currently only 'x', 'y' and 'z' are accepted.
    Length must be equal to the length of p0. If None then all axes will be set to 'x'
    :param lr: Learning rate
    :param print_freq: optional, int, the error will be printed every print_freq iterations. Default = 10.
    :param robust: bool, optional, if set to true the update function will use the jacobian transpose as an alternative
    to the pseudo inverse in case of singularities
    :return:
    np.ndarray, computed joint angles.

    TODO: project angles to -pi <-> pi interval
    """

    def update(J_, e_, target_, lr_):
        try:
            J_inv = compute_jacobian_inverse(J_)
        except np.linalg.LinAlgError as LE:
            if robust:
                J_inv = J_.T
                print("Using transpose!")
            else:
                raise LE

        dtarget = target_ - e_
        return np.matmul(J_inv, dtarget) * lr_

    return base_ik_solver(update,
                          target=target,
                          p0=p0,
                          theta0=theta0,
                          end_eff=end_eff,
                          i_=i_,
                          j_=j_,
                          tol=tol,
                          max_iter=max_iter,
                          rotation_axes=rotation_axes,
                          lr=lr
                          )


def damped_least_squares_solver(target, p0, lambda_, theta0=None, end_eff=None , i_=None, j_=None, tol=1.0e-4,
                                max_iter=10, print_freq=10, rotation_axes=None, lr=1.):
    """ This is an IK solver using the pseudo inverse of the Jacobian to solve the optimisation problem and adds a
    regularisation term.

    :param target: target for the optimisation problem
    :param p0: joint offsets
    :param theta0: initial joint angles
    :param end_eff: end effector position with respect to its frame origin
    :param i_: TODO
    :param j_: TODO
    :param tol: tolerance, when the error drops bellow this number, the optimisation is considered converged.
    :param max_iter: max number of iterations before the optimisation is stopped
    :param rotation_axes: list(str) Rotation axes of the revolute joints, currently only 'x', 'y' and 'z' are accepted.
    Length must be equal to the length of p0. If None then all axes will be set to 'x'
    :param lr: Learning rate
    :param print_freq: optional, int, the error will be printed every print_freq iterations. Default = 10.
    :return:
    np.ndarray, computed joint angles.

    TODO: project angles to -pi <-> pi interval
    """
    lambda__sq_ = lambda_ ** 2

    def update(J_, e_, target_, lr_):
        J_inv = compute_jacobian_inverse(J_, lambda__sq_)
        dtarget = target_ - e_
        return np.matmul(J_inv, dtarget) * lr_

    return base_ik_solver(update,
                          target=target,
                          p0=p0,
                          theta0=theta0,
                          end_eff=end_eff,
                          i_=i_,
                          j_=j_,
                          tol=tol,
                          max_iter=max_iter,
                          rotation_axes=rotation_axes,
                          lr=lr)
