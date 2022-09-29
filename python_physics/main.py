import time

from physicsObject import PhysicsObject, Circle, Box
import pygame
from scene import Scene
import lemkelcp as lcp
from LCPsolvers import pivoting_methods, Jacobi, PGS
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

    obj1 = Circle(pos=[200, 400], idx=0, vel=[3, 0], col=BLUE, rad=50)
    obj2 = Circle(pos=[400, 400], idx=1, vel=[-3, 0], col=GREEN, rad=60)
    obj3 = Circle(pos=[300, 300], idx=2, vel=[0, 4], col=RED, rad=50)
    ground_radius = 300000
    obj4 = Circle(pos=[400, ground_radius + 700], idx=3, vel=[0, 0], col=RED,
        rad=ground_radius, enable_phyics=False)

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
        added = []
        for obj_idx in range(len(scene.objects)):
            for obj_idx2 in range(len(scene.objects)):
                if obj_idx != obj_idx2 and ((obj_idx2, obj_idx) not in added):
                    added.append((obj_idx, obj_idx2))
                    test = scene.objects[obj_idx].intersect(scene.objects[obj_idx2])
                    if test.doesIntersect:
                        # save indices
                        collisions.append(test)

        '''
        We know so far there's two methods 1. Constraint 2. Impulse based.
        We're going to use Impulse based methods, solving them using a
        complementarity problem formulation. Complementarity comes from the
        fact that we insert an impulse based on the gap-function phi hence
        comeplementing it.

        TODO: An important NOTE is that we're not considering friction NOR
        rotational velocities.
        '''

        print("added", added)
        print("collisions", collisions)

        dim = 8
        M = np.eye(dim)
        print(M)
        #J = np.zeros(6)  # num_objs * 2
        J = []
        indices = []
        phi = []
        # phi = num_contacts * 2 (non zero J)

        u = np.zeros(dim)
        grav = np.zeros(dim)
        grav[1] = 0.9
        grav[3] = 0.9
        grav[5] = 0.9
        grav[7] = 0.9

        # go through each collision and fix it!
        for c in collisions:
            obj1, obj2 = c.objs
            indices.append((obj1.idx, obj2.idx, c.normal))
            # this is indeed the mass matrix
            # phi_tmp = c.distance + scene.delta_time * c.contact_vel
            # phi.append(phi_tmp[0])
            # phi.append(phi_tmp[1])

            # The c.contact_vel = J * u can be written as a matrix-vector product
            # J = [-c.normal^T c.normal^T]
            # u = [obj1.vel obj2.vel]   (page 17)

            # [M -J^T   * [u, l]  + [-Mu - hf, 0] = 0   (page 28)
            #  J    0]

            # [J M^-1 J^T] * l + J * M^-1 (Mu + hf) = 0

            # After computing phi it's a matter of computing the impulse
            # for both colliding objects

            # Numerical methods: pivoting methods OR iterative methods
            # each method has trade offs regarding performance, accuracy
            # and robustness

            # 0 <= phi compl. l >= 0
            # 0 <= (phi + h * contact_v) compl. l >= 0 (page 16)
            # 0 <= Ax + b compl. x >= 0
            # where x are the impulses (it could be lambda for just contacts)
            tmp_jac = np.zeros(dim)

            u[(obj1.idx * 2)] = -c.contact_vel[0]
            u[(obj1.idx * 2) + 1] = -c.contact_vel[1]

            u[(obj2.idx * 2)] = c.contact_vel[0]
            u[(obj2.idx * 2) + 1] = c.contact_vel[1]

            tmp_jac[(obj1.idx * 2)] = -c.normal[0]
            tmp_jac[(obj1.idx * 2) + 1] = -c.normal[1]

            tmp_jac[(obj2.idx * 2)] = c.normal[0]
            tmp_jac[(obj2.idx * 2) + 1] = c.normal[1]

            J.append(tmp_jac)  # a single row per contact.

        if len(J) == 0:
            l = 0
        else:

            J = np.array(J)

            print("J", J)
            print("J shape", J.shape)

            A = J @ np.linalg.inv(M) @ np.transpose(J)

            print("A", A)
            print("A shape", A.shape)

            b = J @ np.linalg.inv(M) @ (M @ u)

            # J => num_contactsxDOF DOFxDO DOFxnum_contacts
            # A -> num_contactsxnum_contacts

            # l = lcp.lemkelcp(A, b)
            l = pivoting_methods(A, b, method="principal")
            # l = Jacobi(A, b).x.reshape(-1)
            # l = l[0]
            print("lambda", l)

        idx = 0
        for obj1_idx, obj2_idx, normal in indices:
            obj1 = scene.objects_dict[obj1_idx]
            obj2 = scene.objects_dict[obj2_idx]
            if not obj1.no_physics:
                obj1.velocity += -normal * l[idx]
            if not obj2.no_physics:
                obj2.velocity += normal * l[idx]
            idx += 1

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
