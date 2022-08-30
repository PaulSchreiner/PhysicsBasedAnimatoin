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
        pygame.draw.circle(surface=screen,
            color=self.colour, center=self.position,
            radius=self.radius)

    def intersect(self, otherObj):
        radiusSum = self.radius + otherObj.radius
        centerDistance = np.linalg.norm(otherObj.position - self.position)

        # everything points to otherObj
        normal = (otherObj.position - self.position) / centerDistance
        contact_vel = np.dot(otherObj.velocity - self.velocity, normal) * normal

        return IntersectData(obj1=self, obj2=otherObj,
                intersect=centerDistance < radiusSum,
                dist=centerDistance - radiusSum, normal=normal,
                contact_vel=contact_vel)
