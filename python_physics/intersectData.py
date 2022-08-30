class IntersectData:
    def __init__(self, obj1, obj2, intersect, dist, normal, contact_vel):
        self.objs = (obj1, obj2)
        self.distance = dist
        self.doesIntersect = intersect
        self.normal = normal
        self.contact_vel = contact_vel
