#include "collision_detection/impulse_based.h"
#include <iostream>

int main() {
    RigidBody r1     = RigidBody();
    r1.m_centre      = Vector3d(0,0,0); 
    r1.m_invMass     = 1/10;
    r1.m_linVelocity = Vector3d(1,0,0);
    r1.m_forces      = Vector3d(0.4,0.4,0.4);

    r1.m_invInertia << 1,2,3,4, 5,6,7,8, 9,10,11,12, 13, 14, 15, 16;

    std::cerr << r1.m_invInertia << std::endl;

    return 0;
}