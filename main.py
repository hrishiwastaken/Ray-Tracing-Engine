import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.widgets import TextBox
import math
from abc import ABC, abstractmethod
from typing import List
from matplotlib.path import Path
from matplotlib.patches import PathPatch
# Initializing Matplotlib
fig, ax = plt.subplots(figsize=(8, 8))
box_ax = None
textbox = None
air_refractive_index = 1.0

# Function to reset and format the axes upon every click
def setup_axes():
    ax.set_aspect("equal")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title("Recursive Ray Tracing \n Left Click to move Source, Right Click to place mirror, Shift + Right Click to place inverted mirror, C to undo mirror, Control + Right Click to place lens.")

class Ray:
    def __init__(self,origin,direction,surfaces,bounces=10,color="orange",in_glass=False):
        self.origin = np.array(origin, dtype=float)
        self.direction = direction / np.linalg.norm(direction)
        self.length = 100
        self.surfaces: List[Surface] = surfaces
        self.bounces = bounces #---Ray Bounce Limit to prevent infinite bouncing
        self.color = color     #---Ray Colour
        self.in_glass = in_glass

        # Default parametric ray equation
        self.end = self.origin + self.length * self.direction
        # Checking if Ray intersects with any Surface
        self.points = []
        for surface in self.surfaces:
            # Aperture check for each mirrors
            t_intersection = surface.ray_intersection(self)
            if not t_intersection:
                pass
            else:
                self.points.append((t_intersection,surface))
        # Bounce check
        if self.bounces <= 0:
            pass
        else:
            if self.points != []:
                t,surface = min(self.points, key=lambda x: x[0])  # Finds point of intersection closest to ray, i.e the surface which the ray intersects first, to prevent ghost rays.
                self.end = self.origin *t*self.direction
                self.draw()
                self.draw_reflection_or_refraction(surface)
            else:
                self.draw()
            return

    def draw(self):
        z = 2 if self.color == "blue" else 1
        l = 2 if self.color == "blue" else 1.5
        ax.plot(
            [self.origin[0], self.end[0]],
            [self.origin[1], self.end[1]],
            color=self.color,
            alpha=0.8,
            linewidth=l,
            zorder=z
        )

    def draw_reflection_or_refraction(self, surface):
        incident_dir = self.direction
        normal = surface.normal(self.end)
        normal /= np.linalg.norm(normal)

        # Standard Mirror Reflection
        if not surface.is_lens:
            dot = np.dot(incident_dir,normal)
            reflected_dir = incident_dir -2*dot*normal

            Ray(
                origin=self.end,
                direction=reflected_dir,
                surfaces=self.surfaces,
                bounces=self.bounces-1,
                color="blue",
                )

        # Standard Lens Refraction (Snell's Law)
        else:
            dot = np.dot(incident_dir, normal)

            # Determine whether entering or exiting glass
            if dot < 0:
                oriented_normal = normal
            else:
                oriented_normal = -normal
            if not self.in_glass:
                n1,n2 = air_refractive_index, surface.refractive_index
            else:
                n1,n2 = surface.refractive_index, air_refractive_index

            n = n1 / n2
            cos_i = -np.dot(incident_dir, oriented_normal)  # The main dot product
            sin2_t = (n**2) * (1.0 - cos_i**2)              # Used for checking total internal reflection(negative root)

            # Total Internal Reflection check
            if sin2_t > 1.0:
                refracted_dir = (
                    incident_dir -2*(-cos_i)*oriented_normal
                )
                glasscheck = self.in_glass
            else:
                cos_t = math.sqrt(1.0-sin2_t)
                refracted_dir = (
                    n * incident_dir + (n * cos_i - cos_t) * oriented_normal
                )
                glasscheck = not self.in_glass

                Ray(
                    origin=self.end,
                    direction=refracted_dir,
                    surfaces=self.surfaces,
                    bounces=self.bounces-1,
                    color="blue",
                    in_glass = glasscheck,
                )

# NEW
# Surface Parent Class
# Contains core properties and abstract methods that exist in every surface no matter what.
# Note: I am classifying Concave as opens left and Convex as opens right for singular surfaces.  Lens nomenclature is as is(however the previous mentioned nomenclature is used internally)

class Surface(ABC):
    def __init__(self,centre,hyperparameter,is_concave,parameter=None,is_lens=False,refractive_index=1.5):
        self.centre = np.array(centre, dtype=float)
        self.is_lens = is_lens
        self.is_concave = is_concave
        if parameter is None:
            parameter = []
        if self.is_lens:
            self.refractive_index = refractive_index

    @abstractmethod
    def normal(self,point: np.ndarray) -> np.ndarray:
        """Returns normal vector at a given point on a surface."""
        pass

    @abstractmethod
    def ray_intersection(self,ray: Ray) -> List[np.ndarray]:
        """Returns all intersection points of given ray with surface."""
        pass

    @abstractmethod
    def surface_intersection(self,surface) -> List[float]:
        """Finds parametric values of intersection with another given surface of the same type for lens construction using hyperparameters of second surface."""
        pass

    @abstractmethod
    def draw(self):
        """Draws the surface on graph as per its mathematical nature."""
        pass

    
# Now, specific geoemtries of different surfaces can be incorporated into a more generalized framework using Polymorphism.

class Circle(Surface):
    def __init__(self,centre,hyperparameter,is_concave,parameter=None,is_lens=False,refractive_index=1.5):
        super().__init__(centre,hyperparameter,is_concave,parameter,is_lens,refractive_index)
        self.radius = hyperparameter
        self.centre_angle = 0 if self.is_concave else 180

    def normal(self,point: np.ndarray) -> np.ndarray:
        """Returns normal vector at a given point on a surface."""
        return np.array([2*(point[0]-self.centre[0]),2*(point[1]-self.centre[1])]) 

    def ray_intersection(self,ray: Ray) -> List[np.ndarray]:
        """Returns all intersection points of given ray with surface."""
        ray_origin = ray.origin
        f = ray_origin[0] - self.centre[0]
        g = ray_origin[1] - self.centre[1]

        Beta = f * ray.direction[0] + g * ray.direction[1]
        Delta = f**2 + g**2 - self.radius**2
        discriminant = 4 * (Beta**2) - 4 * Delta

        if discriminant > 0:             # Using Quadratic Discriminant to check for intersection
            t1 = (-2 * Beta + discriminant**0.5) / 2
            t2 = (-2 * Beta - discriminant**0.5) / 2
            valid_t = sorted([t for t in (t1,t2) if t > 1e-5])
            for pt in valid_t:
                intersection_point = ray.origin + pt * ray.direction
                r = intersection_point - self.centre
                angle_deg = math.degrees(math.atan2(r[1], r[0]))
                diff = (angle_deg - self.centre_angle + 180) % 360 -180
                if min(self.parameter) <= diff <= max(self.parameter):
                    return [pt]
        return []

    def surface_intersection(self, surface: Surface):
        """Finds parametric values of intersection with another given surface of the same type for lens construction using hyperparameters of second surface."""
        surface_centre = surface.centre
        surface_radius = surface.radius
        g = self.centre[0] - surface_centre[0]
        A = 2*(self.centre[1] - surface_centre[1])
        B = g**2 + surface_centre[1]**2 - self.centre[1]**2 + self.radius**2 - surface_radius**2
        alpha = A**2 + 4*(g**2)
        beta = 2*A*B - 8*(g**2)*self.centre[1]
        gamma = B**2 - 4*(g**2)*(self.radius**2) + 4*(g**2)*(self.centre[1]**2)
        D = beta**2 - 4*alpha*gamma
        ys = [(-beta + D**0.5)/(2*alpha), (-beta - D**0.5)/(2*alpha)]

        angles = []
        for y in ys:
            x = self.centre[0] - (A*y+B)/(2*g)
            angle = math.degrees(math.atan2(y-self.centre[1], x-self.centre[0]))
            diff = (angle-self.centre_angle+180)%360-180
            angles.append(diff)
        

        self.parameter = sorted(angles)  

    def get_coordinate(self, parameter):
        return np.array([self.centre[0] + self.focus*(parameter**2),self.centre[2] + 2*self.focus*parameter])
    
    def draw(self):
        """Draws the surface on graph as per its mathematical nature."""
        t1 = (self.centre_angle + self.parameter[0]) % 360
        t2 = t1 + (self.parameter[1] - self.parameter[0])
        arc = Arc(
            tuple(self.centre), width=2*self.radius, height=2*self.radius,
            angle=0, theta1=t1, theta2=t2,
            linewidth=3, color='black'
        )
        ax.add_patch(arc)

class Parabola(Surface):
    def __init__(self,centre,hyperparameter,is_concave,parameter=None,is_lens=False,refractive_index=1.5):
        super().__init__(centre,hyperparameter,is_concave,parameter,is_lens,refractive_index)
        self.focus = hyperparameter
        self.orientation = -1 if is_concave else 1
    def normal(self,point: np.ndarray) -> np.ndarray:
        """Returns normal vector at a given point on a surface."""
        return np.array([-4*self.orientation*self.focus, 2*(point[1]-self.centre[1])])

    def ray_intersection(self,ray: Ray) -> List[np.ndarray]:
        """Returns all intersection points of given ray with surface."""
        f = ray.origin[1] - self.centre[1]
        g = ray.origin[0] - self.centre[0]
        A = ray.direction[1]**2
        B = (2*f*ray.direction[1] + 4*self.focus*ray.direction[0]) if self.is_concave else (2*f*ray.direction[1] - 4*self.focus*ray.direction[0]) 
        C = (4*self.focus*g + f**2) if self.is_concave else (f**2 - 4*self.focus*g)
        D = B**2 - 4*A*C
        if A != 0:
            if D > 0:
                t1 = (-B + D**0.5) / (2*A)
                t2 = (-B - D**0.5) / (2*A)
                valid_t = sorted([t for t in (t1,t2) if t > 1e-5])
                for pt in valid_t:
                    intersection_point = ray.origin + pt*ray.direction
                    param = self.get_parameter(intersection_point)
                    if min(self.parameter) <= param <= max(self.parameter):
                        return [pt]
        else:
            if B != 0:
                pt = -C/B
                if pt > 1e-5:
                    intersection_point = ray.origin + pt*ray.direction
                    param = self.get_parameter(intersection_point)
                    if min(self.parameter) <= param <= max(self.parameter):
                        return [pt]
            
        return []

    def surface_intersection(self, surface: Surface):
        surface_centre = surface.centre
        surface_focus = surface.focus
        s1 = self.orientation
        s2 = surface.orientation
        f1 = self.focus
        f2 = surface_focus
        h1, k1 = self.centre
        h2, k2 = surface_centre
        A = s2*f2 - s1*f1
        B = 2*(s1*f1*k2 - s2*f2*k1)
        C = (
            s2*f2*k1**2
            - s1*f1*k2**2
            + 4*s1*s2*f1*f2*(h1-h2)
        )
        discriminant = B**2 - 4*A*C
        if discriminant > 0:
            D = math.sqrt(discriminant)
            roots = [
                (-B+D)/(2*A),
                (-B-D)/(2*A)
            ]
            for root in roots:
                self.parameter.append(
                    (root-k1)/(2*s1*f1)
                )
            self.parameter = sorted(self.parameter)

    def get_parameter(self, point):
        return (point[1] - self.centre[1]) / (2 * self.orientation * self.focus)

    def get_coordinate(self, parameter):
        return np.array([self.centre[0] + self.focus*(parameter**2),self.centre[2] + 2*self.focus*parameter])
    
    
    def draw(self):
        t = np.linspace(min(self.parameter), max(self.parameter), 500)
        x = self.centre[0] + self.orientation*self.focus*t**2
        y = self.centre[1] + 2*self.orientation*self.focus*t

        points = np.column_stack((x, y))
        path = Path(points)
        patch = PathPatch(path, fill=False, linewidth=3, color='black')
        ax.add_patch(patch)

class Line(Surface):
    def __init__(self,centre,hyperparameter,is_concave,parameter=None,is_lens=False,refractive_index=1.5):
        super().__init__(centre,hyperparameter,is_concave,parameter,is_lens,refractive_index)
        self.origin = self.centre
        self.direction = hyperparameter
        self.length = parameter
        self.end = self.origin + self.length*self.direction

        def normal(self,point: np.ndarray = []) -> np.ndarray:
            """Returns normal vector at a given point on a surface."""
            return np.array([-self.direction[1],self.direction[0]])

        def ray_intersection(self,ray: Ray) -> List[np.ndarray]:
            """Returns all intersection points of given ray with surface."""
            p1 = ray.origin
            p2 = self.end
            delX = p2[0] - p1[0]; delY = p2[1] - p1[1]
            phi = math.atan2(self.direction[1], self.direction[0])
            theita = math.atan2(ray.direction[1], ray.direction[0])
            denom = math.sin(phi - theita)
            if abs(denom) < 1e-10:
                return []
            t = (delX*math.sin(phi) -delY*math.cos(phi))/denom
            if t <= 1e-5:
                return []

            intersection = ray.origin + t*ray.direction
            segmentV = p2 - self.origin
            intersectionV = intersection - self.origin
            projection = np.dot(intersectionV, segmentV)
            if 0 <= projection <= np.dot(segmentV, segmentV):
                return [t]
            return []

        def surface_intersection(self,surface) -> List[float]:
            """Finds parametric values of intersection with another given surface of the same type for lens construction using hyperparameters of second surface."""
            pass
            
        def draw(self):
            """Draws the surface on graph as per its mathematical nature."""
            self.color = "black"
            self.lw = 3
            ax.plot(
                [self.origin[0], self.end[0]],
                [self.origin[1], self.end[1]],
                color=self.color,
                alpha=0.8,
                linewidth=self.lw,
                zorder=3
            )

class Lens():
    def __init__(self,origin,hp1,hp2,thickness,lenstype):
        self.origin = np.array(origin, dtype=float)
        self.thickness = thickness
        self.conic = self.conic
        self.hp1 = hp1
        self.hp2 = hp2
        self.lenstype = lenstype
        self.lenstypes = {
            "parabolic": Parabola,
            "circular":  Circle 
        }
        self.lens_obj = []
    def draw_convex(self):
        centre1 = [(self.origin[0] - (self.thickness/2)), self.origin[1]]
        centre2 = [(self.origin[0] + (self.thickness/2)), self.origin[1]]  # In case i want to add non-horizontal axial lenses, this is where the logic would start from, because the rest is generaized
        surface1: Surface = self.lenstypes[self.lenstype](centre1,self.hp1,False,True)
        surface2: Surface = self.lenstypes[self.lenstype](centre2,self.hp2,True,True)
        self.lens_obj.extend([surface1,surface2])
        # Parametrizes the surfaces and internalizes boundary parameters to the respective surface.
        surface1.surface_intersection(surface2)
        surface2.surface_intersection(surface1)
        self.draw_lens()
        return self.lens_obj
    
    def draw_concave(self):
        centre1 = [(self.origin[0] - (self.thickness/2)), self.origin[1]]
        centre2 = [(self.origin[0] + (self.thickness/2)), self.origin[1]]  # In case i want to add non-horizontal axial lenses, this is where the logic would start from, because the rest is generaized
        surface1: Surface = self.lenstypes[self.lenstype](centre1,self.hp1,True,True)
        surface2: Surface = self.lenstypes[self.lenstype](centre2,self.hp2,False,True)
        self.lens_obj.extend([surface1,surface2])
        endpoints1 = [surface1.get_coordinate(sorted(surface1.parameter)[0]), surface1.get_coordinate(sorted(surface1.parameter)[1])]
        endpoints2 = [surface2.get_coordinate(sorted(surface2.parameter)[0]), surface2.get_coordinate(sorted(surface2.parameter)[1])]
        length = np.linalg.norm(endpoints2[1]-endpoints1[1])
        direction = (endpoints2[1][1]-endpoints1[1][1])/(endpoints2[0][0]-endpoints1[0][0])
        surface3 = Line(endpoints1[0],direction,True,length,True)
        self.lens_obj.append(surface3)
        surface1.surface_intersection(surface2)
        surface2.surface_intersection(surface1)
        self.draw_lens()
        return self.lens_obj
    
    def draw_lens(self):
        for surface in self.lens_obj:
            surface.draw()
            
            
class Source:
    def __init__(self,origin, surfaces: Surface, rays):
        self.rays = []
        self.origin = np.array(origin, dtype=float)
        self.surfaces = surfaces
        n = 360/rays

        for i in range(rays):
            a = np.radians(i*n)
            d = np.array([np.cos(a), np.sin(a)])
            self.rays.append(Ray(self.origin, d, self.mirrors))

        ax.scatter(self,origin[0], self.origin[1], color="red", zorder=5, label="Source")


# Initializing Default Surface
Surfaces = []
source_pos = np.array([10.0, 10.0])