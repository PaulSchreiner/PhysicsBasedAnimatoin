#include <collision_detection/impulse_based.h>

void AddCollisionImpulse (RigidBody & c0,
			                    RigidBody & c1, 
                          const Vector3d & hitPoint,
                          const Vector3d & normal,
                          float penetration)
{
  float invMass0 = c0.m_invMass;
  float invMass1 = c1.m_invMass;

  const Matrix4d& worldInvInertia0 = c0.m_worldInvInertia;
  const Matrix4d& worldInvInertia1 = c1.m_worldInvInertia;

  if ((invMass0 + invMass1) == 0.0) return; 

  Vector3d r0 = hitPoint - c0.m_centre;
  Vector3d r1 = hitPoint - c1.m_centre;

  Vector3d v0 = c0.m_linVelocity + c0.m_angVelocity.cross(r0);
  Vector3d v1 = c1.m_linVelocity + c1.m_angVelocity.cross(r1);

  Vector3d dv = v0 - v1;
  
  float relativeMovement = -dv.dot(normal);
  if (relativeMovement < - 0.01f)
  {
    return;
  }
  {
    float e = 0.0f;

    Vector3d vec1 = r0.cross(normal);

    // float normDiv = normal.dot(normal) * ((invMass0 + invMass1);
  }
}
