from intersectData import IntersectData
import numpy as np
import pygame


class PhysicsObject:

    def __init__(self, pos, vel, col, rad, enable_phyics=True):
        self.position = np.array(pos, dtype=float)
        self.velocity = np.array(vel, dtype=float)
        self.colour = col
        self.radius = rad
        self.no_physics = not enable_phyics

    def draw(self, screen):
        pygame.draw.circle(screen,
                           self.colour,
                           [int(f) for f in self.position],
                           self.radius)

    def intersect(self, other_obj):
        radius_sum = self.radius + other_obj.radius
        center_distance = np.linalg.norm(other_obj.position - self.position)

        # everything points to other_obj
        normal = (other_obj.position - self.position) / center_distance
        contact_vel = np.dot(other_obj.velocity - self.velocity, normal) * normal

        # jac = np.array([[-normal[0], 0, normal[0], 0],
        #                 [0, -normal[1], 0, normal[1]]])

        jac = np.array([-normal[0], -normal[1], normal[0], normal[1]])  # row
        jac2 = np.array([normal[0], normal[1], -normal[0], -normal[1]])
        diff = np.array([self.velocity[0], self.velocity[1],
                other_obj.velocity[0], other_obj.velocity[1]])  # column

        contact_vel2 = np.matmul(jac, diff) * normal
        print("dot prod: ", np.dot(other_obj.velocity - self.velocity, normal))
        print("normal:", normal)
        print("jacobian:", np.matmul(jac, diff))
        print("contact_vel", contact_vel)
        print("contact_vel2", contact_vel2)
        print("##########")

        test = np.matmul(jac, diff)
        return IntersectData(obj1=self, obj2=other_obj,
                intersect=center_distance < radius_sum,
                dist=center_distance - radius_sum, normal=normal,
                contact_vel=contact_vel2, jac=jac, jac2=jac2)
