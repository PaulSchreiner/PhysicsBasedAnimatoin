import time

from physicsObject import PhysicsObject
import pygame
from scene import Scene
import lemkelcp.lemkelcp as lcp
from LCPsolvers import incremental_pivoting
import numpy as np

BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)


def main():
    background_colour = (255, 255, 255)
    (width, height) = (800, 800)
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Dumb physics example')
    screen.fill(background_colour)
    pygame.display.flip()
    running = True

    scene = Scene(dt=0.1, acceleration=[0, 0.9])

    obj1 = PhysicsObject(pos=[200, 400], vel=[3, 0], col=BLUE, rad=50)
    obj2 = PhysicsObject(pos=[400, 400], vel=[-3, 0], col=GREEN, rad=60)
    obj3 = PhysicsObject(pos=[300, 300], vel=[0, 4], col=RED, rad=50)
    ground_radius = 300000
    obj4 = PhysicsObject(pos=[400, ground_radius + 700], vel=[0, 0], col=RED, rad=ground_radius, enable_phyics=False)

    scene.addObject(obj1)
    scene.addObject(obj2)
    scene.addObject(obj3)
    scene.addObject(obj4)

    while running:
        screen.fill(background_colour)

        # integrate time in the scene
        scene.integrate()

        # collision detection
        collisions = []
        for obj_idx in range(len(scene.objects)):
            for obj_idx2 in range(len(scene.objects)):
                if obj_idx != obj_idx2:
                    test = scene.objects[obj_idx].intersect(scene.objects[obj_idx2])
                    if test.doesIntersect:
                        collisions.append(test)
                        # print("Collision!!")

        # go through each collision and fix it!
        # we know so far there's two methods 1. Constraint 2. Impulse based.
        for c in collisions:
            obj1, obj2 = c.objs

            M = np.eye(2)

            phi = c.distance + scene.delta_time * c.contact_vel

            # print("phi", phi)

            ######################## Implement LCP solver here ##############################
            # t0 = time.time()
            # l, exit_code, exit_string = lcp.lemkelcp(M, phi)
            # t1 = time.time()
            l = incremental_pivoting(M, phi)
            # t2 = time.time()
            # print("lemke time: {}".format(t1-t0))
            # print("our time: {}".format(t2-t1))
            # print("Validation lcp solution: l = {}".format(l))
            # print("Experimental lcp solution: l = {}".format(l_test))

            ######################## Implement LCP solver here ##############################

            if not obj1.no_physics:
                obj1.velocity += -c.normal * l
            if not obj2.no_physics:
                obj2.velocity += c.normal * l

        # scene.applyContacts()

        for obj in scene.objects:
            obj.draw(screen)

        pygame.display.update()

        # check for events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            keys = pygame.key.get_pressed()

            if np.any(keys):
                scene.reset()

if __name__ == '__main__':
    main()
