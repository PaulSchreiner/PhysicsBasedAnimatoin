from physicsObject import PhysicsObject
import pygame
from scene import Scene
import lemkelcp as lcp
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

    scene.addObject(obj1)
    scene.addObject(obj2)
    scene.addObject(obj3)

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
            M = np.eye(2)
            phi = c.distance + scene.delta_time * c.contact_vel

            # print("phi", phi)

            l, exit_code, exit_string = lcp.lemkelcp(M, phi)
            obj1, obj2 = c.objs

            obj1.velocity += -c.normal * l
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
