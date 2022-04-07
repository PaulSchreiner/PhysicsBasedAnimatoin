#include "geometry/rigid_body.h"

static
  void AddCollisionImpulse (RigidBody & c0,
			                RigidBody & c1, 
                            const Vector3d & hitPoint,
                            const Vector3d & normal,
                            float penetration);
