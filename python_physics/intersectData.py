class IntersectData:
    def __init__(self, obj1, obj2, intersect, dist, normal, contact_vel, jac,
            jac2):
        self.objs = (obj1, obj2)
        self.distance = dist
        self.doesIntersect = intersect
        self.normal = normal
        self.contact_vel = contact_vel
        self.jac = jac
        self.jac2 = jac2
