#include <eigen3/Eigen/Dense>

using
  Eigen::Matrix4d;
using
  Eigen::Vector3d;
using
  Eigen::Quaterniond;

class
  RigidBody
{
public:
  // LINEAR
  Vector3d m_centre;
  float m_invMass;
  Vector3d m_linVelocity;
  Vector3d m_forces;

  // ANGULAR
  Matrix4d m_invInertia;
  Vector3d m_angVelocity;
  Quaterniond m_orientation;
  Vector3d m_torques;

  // Combined
  Matrix4d m_matWorld;
  Matrix4d m_worldInvInertia;
};
