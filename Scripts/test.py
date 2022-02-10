from IK.InverseKinematics import pseudo_inverse_solver
import numpy as np


'''Test with a single joint:'''
p = np.array([[0., 1., 0.], [0., 1., 0.], [0., 0., 0.]])

target = np.array([0., 2., 0., 1.])

theta0 = np.random.uniform(0, np.pi, p.shape[0])
print("Initial joint angles: {}".format(theta0))

rotation_axes = ['x', 'y', 'z']
rotation_axes = None
theta = pseudo_inverse_solver(target, p, theta0=theta0, rotation_axes=rotation_axes, lr=0.5, max_iter=100)

print("Computed joint angles: {}".format(theta))
