from IK.InverseKinematics import *
import numpy as np


'''Test with three links and an offset end effector:'''
# Define the links by setting some offset parameters (the last link is the end effector)
p = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
# p = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])

end_eff = np.array([0., 0., 0., 1.])
# i_vec = np.array([1., 0., 0., 0.])
# j_vec = np.array([0., 1., 0., 0.])
i_vec = None
j_vec = None

# Set a target
# target = np.array([1., 0., 0., 1., 0., 0., 0., 1., 0.])
target = np.array([1., 0., 0.])

# Initialise the joint angles
theta0 = np.random.uniform(-np.pi, np.pi, p.shape[0])
# theta0 = np.zeros(p.shape[0])
print("Initial joint angles: {}".format(theta0))

# Define the rotation axes of the joint (None means all are revolute around x-axis)
# rotation_axes = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
rotation_axes = ['x', 'y', 'z', 'x', 'y', 'z']
# rotation_axes = ['x', 'y', 'z']
# rotation_axes = None

# And go!
theta_pi = pseudo_inverse_solver(target,
                                 p,
                                 end_eff=end_eff,
                                 i_=i_vec,
                                 j_=j_vec,
                                 theta0=theta0.copy(),
                                 rotation_axes=rotation_axes,
                                 lr=1,
                                 max_iter=100,
                                 print_freq=10
                                 )

theta_dls = damped_least_squares_solver(target,
                                        p,
                                        end_eff=end_eff,
                                        i_=i_vec,
                                        j_=j_vec,
                                        lambda_=0.001,
                                        theta0=theta0.copy(),
                                        rotation_axes=rotation_axes,
                                        lr=1,
                                        max_iter=100,
                                        print_freq=10
                                        )


print("Computed joint angles using pseudo inverse: {}".format(theta_pi))
print("Computed joint angles using damped least squares: {}".format(theta_dls))
