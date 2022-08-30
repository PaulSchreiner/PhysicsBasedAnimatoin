import copy
import numpy as np


class Scene:

    def __init__(self, dt, acceleration):
        self.initial_objs = []
        self.objects = []
        self.delta_time = dt
        self.acceleration = np.array(acceleration)

    def addObject(self, obj):
        self.objects.append(obj)
        self.initial_objs.append(copy.deepcopy(obj))

    def integrate(self):
        for obj in self.objects:
            obj.velocity += self.acceleration * self.delta_time
            obj.position += obj.velocity * self.delta_time

    def reset(self):
        print("Resetting")
        self.objects = copy.deepcopy(self.initial_objs)
