import abc

from intersectData import IntersectData
import numpy as np
import pygame


class PhysicsObject:

    def __init__(self, pos, vel, col, rad, dims, idx, enable_phyics=True):
        self.position = np.array(pos, dtype=float)
        self.velocity = np.array(vel, dtype=float)
        self.colour = col
        self.radius = rad
        self.no_physics = not enable_phyics
        self.idx = idx
        self.dims = dims

    @abc.abstractmethod
    def draw(self, screen):
        raise NotImplementedError

    @abc.abstractmethod
    def intersect(self, other_obj):
        raise NotImplementedError


class Circle(PhysicsObject):

    def __init__(self, pos, vel, col, rad, idx, enable_phyics=True):
        super().__init__(pos=pos, vel=vel, col=col,
            rad=rad, dims=None, idx=idx, enable_phyics=enable_phyics)

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

        return IntersectData(obj1=self, obj2=other_obj,
                intersect=center_distance < radius_sum,
                dist=center_distance - radius_sum, normal=normal,
                contact_vel=contact_vel)


class Box(PhysicsObject):

    def __init__(self, pos, vel, col, dims, idx, enable_phyics=True):
        super().__init__(pos=pos, vel=vel, col=col,
            rad=None, dims=dims, idx=idx, enable_phyics=enable_phyics)

    def draw(self, screen):

        rect = pygame.Rect([int(f) for f in self.position],
                           [int(f) for f in self.dims])
        pygame.draw.rect(surface=screen,
                         color=self.colour,
                         rect=rect)

    def intersect(self, other_obj):
        # radius_sum = self.radius + other_obj.radius
        # center_distance = np.linalg.norm(other_obj.position - self.position)

        # # everything points to other_obj
        # normal = (other_obj.position - self.position) / center_distance
        # contact_vel = np.dot(other_obj.velocity - self.velocity, normal) * normal

        # box1 = (x:(xmin1,xmax1),y:(ymin1,ymax1))
        # box2 = (x:(xmin2,xmax2),y:(ymin2,ymax2))
        # isOverlapping2D(box1,box2) = isOverlapping1D(box1.x, box2.x) and
        #                              isOverlapping1D(box1.y, box2.y)

        return IntersectData(obj1=self, obj2=other_obj,
                intersect=False,
                dist=0, normal=[0, 0],
                contact_vel=[0, 0])
