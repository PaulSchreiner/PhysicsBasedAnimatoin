from IK.InverseKinematics import pseudo_inverse_solver
import numpy as np


'''Test with three links:'''
# Define the links by setting some offset parameters (the last link is the end effector)
p = np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

# Initialise the joint angles
theta0 = np.random.uniform(0, np.pi, p.shape[0])
print("Initial joint angles: {}".format(theta0))

# Define the rotation axes of the joint (None means all are revolute around x-axis)
# rotation_axes = ['x', 'y', 'z']
rotation_axes = None

# Set a target
target = np.array([0., 2., 0., 1.])

# And go!
theta = pseudo_inverse_solver(target, p, theta0=theta0, rotation_axes=rotation_axes, lr=0.5, max_iter=100)

print("Computed joint angles: {}".format(theta))
