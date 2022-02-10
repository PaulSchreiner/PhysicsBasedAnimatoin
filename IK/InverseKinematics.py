"""
This file contains an inverse kinematics solver library
"""
import numpy as np

from IK.Jacobian import compute_jacobian_revolute_chain, compute_jacobian_inverse


def pseudo_inverse_solver(target, p0, theta0=None, end_eff=None, tol=1.0e-4, max_iter=10, rotation_axes=None, lr=1.):
    """ This is an IK solver using the pseudo inverse of the Jacobian to solve the optimisation problem. In case of
    singularities (fx. gimbal lock), it will switch to using the Jacobian transpose instead. That last step is a bit
    hacky, but it kinda works to avoid gimbal locks.

    :param target: target for the optimisation problem
    :param p0: joint offsets
    :param theta0: initial joint angles
    :param end_eff: end effector position with respect to its frame origin
    :param tol: tolerance, when the error drops bellow this number, the optimisation is considered converged.
    :param max_iter: max number of iterations before the optimisation is stopped
    :param rotation_axes: list(str) Rotation axes of the revolute joints, currently only 'x', 'y' and 'z' are accepted.
    Length must be equal to the length of p0. If None then all axes will be set to 'x'
    :param lr: Learning rate
    :return:
    np.ndarray, computed joint angles.

    TODO: project angles to -pi <-> pi interval
    """
    if theta0 is None:
        theta0 = [0 for _ in p0]

    if end_eff is None:
        end_eff = np.array([0., 0., 1., 1.])

    theta = theta0

    for ii in range(max_iter):
        J, e = compute_jacobian_revolute_chain(theta, p0, end_eff=end_eff, rotation_axes=rotation_axes)
        err = np.linalg.norm(target - e)
        print("Error: {}".format(err))
        if tol > err:
            print("Converged after {} iterations! Breaking loop.".format(ii))
            return theta
        try:
            J_inv = compute_jacobian_inverse(J)
        except np.linalg.LinAlgError:
            J_inv = J.T
        dtarget = target - e
        dTheta = np.matmul(J_inv, dtarget) * lr
        theta += dTheta

    print("Max iterations reached, did not converge!")
    return theta
