import numpy as np
import math

def simple_region(theta, direction = 'clockwise'):
    region = [[theta, theta + math.pi]] if direction == 'counterclockwise' else [[theta - math.pi, theta]]
    if region[0][0] < - math.pi:
        region = [[- math.pi, region[0][1]], [region[0][0] + 2 * math.pi, math.pi]]
    elif region[0][1] > math.pi:    
        region = [[region[0][0], math.pi], [- math.pi, region[0][1] - 2 * math.pi]]

    return region

def is_in_simple_region(theta, simple_region):
    in_ = False
    for region in simple_region:
        if (theta >= region[0]) and (theta <= region[1]):
            in_ = True
    return in_

class Region:
    """A Region must define a point_project method, which takes in a point and return its projected point."""
    def __init__(self):
        pass
    def point_project(self,pt):
        print('point_project is not defined')
        return None
    
    def point_project_sideway(self, my_loc, pt):
        print('point_project_sideway is not defined')
        return None
    
    def is_inside(self, pt):
        print('is_inside is not defined')
        return None

class RegionsIntersection(Region):
    """The intersection of multiple regions. The projection method is to call all the project_point() methods in sequence."""
    def __init__(self,regions):
        self.regions = regions

    def project_point(self,pt):
        for rg in self.regions:
            pt = rg.project_point(pt)
        return pt
    
    def project_point_sideway(self, pt, my_loc):
        other_theta = []
        current_pt = pt.copy()
        theta = None
        for i, rg in enumerate(self.regions):
            if rg.is_inside(current_pt):
                current_pt = rg.point_project_sideway(my_loc)
                theta = math.atan2(rg.origin[1] - my_loc[1], rg.origin[0] - my_loc[0])
                break
        if theta is None:
            return current_pt
        else:
            theta2 = math.atan2(pt[1] - my_loc[1], pt[0] - my_loc[0])
            # direction = 'clockwise' if theta2 <= theta else 'counterclockwise'
            direction = 'clockwise'
            theta_region = simple_region(theta, direction=direction)
            for i, rg in enumerate(self.regions):
                theta = math.atan2(rg.origin[1] - my_loc[1], rg.origin[0] - my_loc[0])
                if not is_in_simple_region(theta, theta_region):
                    continue
                if not rg.is_inside(current_pt):
                    continue
                else:
                    current_pt = rg.point_project_sideway(my_loc, direction=direction)
            return current_pt            


class Rect2D(Region):
    """A 2-D Rectangle"""
    def __init__(self, xlims=(0,0), ylims=(0,0)):
        super(Rect2D,self).__init__()
        self.xmin = np.min(xlims)
        self.xmax = np.max(xlims)
        self.ymin = np.min(ylims)
        self.ymax = np.max(ylims)

    def project_point(self,pt):
        def constrain(input, low, high):
            # if input < low:
            #     input = low
            # elif input > high:
            #     input = high
            # else:
            #     input = input
            
            return (input<low)*low + (input>high)*high + (np.logical_and(input>=low,input<=high))*input

        pt = np.array(pt)

        if len(pt.shape)==1:
            return np.array([constrain(pt[0],self.xmin,self.xmax),\
                             constrain(pt[1],self.ymin,self.ymax)])
        else:
            return np.hstack([constrain(pt[:,0],self.xmin,self.xmax).reshape(-1,1),\
                constrain(pt[:,1],self.ymin,self.ymax).reshape(-1,1)])

class CircleInterior(Region):
    """CircleInterior"""
    def __init__(self, origin, radius):
        super(CircleInterior, self).__init__()
        self.origin = origin
        self.radius = radius

    def project_point(self,pt):	
        dist=np.linalg.norm(pt-self.origin)

        if dist <= self.radius: # If pt is within the interior of the circle, do nothing
            proj=pt
        else: # If pt goes outside of the circle's interior, project it back to the boundary
            proj=((pt-self.origin)/dist * self.radius).T + self.origin
        return proj


class CircleExterior(Region):
    """CircleExterior. The only difference is a change of inequality direction in the project point method."""
    def __init__(self, origin, radius):
        super(CircleExterior, self).__init__()
        self.origin = np.array(origin)
        self.radius = radius

    def project_point(self,pt):	
        dist=np.linalg.norm(pt-self.origin,axis = -1).reshape(-1,1)
#         if dist >= self.radius: # If pt is outside of the circle, do nothing
#             proj=pt
#         elif dist<self.radius and dist!=0: # If pt goes inside of the circle's interior, project it back to the boundary
#             proj=((pt-self.origin)/dist * self.radius) + self.origin
#         else: # If pt coincides with the origin, project it to the (1,1,1,...) direction on the boundary.
#             one =  np.ones(len(self.origin.ravel()))
#             proj = one/np.linalg.dist(one) * self.radius
        
        # The vectorized implementation of the projection rule above
        def random_proj():
            one =  np.ones(len(self.origin.ravel()))
            proj = self.radius * one/np.linalg.norm(one)
            return proj
        
        proj = (dist>=self.radius)*pt \
            + np.logical_and(dist<self.radius, dist>0)*(((pt-self.origin)/(dist+1e-9) * self.radius) + self.origin)\
            + (dist==0).reshape(-1,1)*random_proj()
        
        return proj
    
    def point_project_sideway(self, my_loc, direction='clockwise'):
        my_proj = self.project_point(np.atleast_2d(my_loc)).squeeze()
        angle = math.atan2(my_proj[1] - self.origin[1], my_proj[0] - self.origin[0])
        if direction == 'clockwise':
            angle += 1/2 * math.pi
        else:
            angle -= 1/2 * math.pi
        
        proj = [self.origin[0] + self.radius * np.cos(angle), self.origin[1] + self.radius * np.sin(angle)]

        return proj

    def is_inside(self, pt):
        return np.linalg.norm(pt - self.origin) < self.radius

    



