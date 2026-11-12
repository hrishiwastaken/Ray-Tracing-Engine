import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, PathPatch
from matplotlib.path import Path
from matplotlib.widgets import Button, TextBox
import math
from abc import ABC, abstractmethod
from typing import List

# Global Physics & App State 
air_refractive_index = 1.0
glass_refractive_index = 1.5
hyperparameter_val = 3.0 
hyperparameter_val2 = 3.0
thickness_val = 1.0
concave_aperture_val = 0.0  # 0.0 = Auto-calculate from convex intersection, or custom span
angle_val = 0.0             # In degrees: positive = anti-clockwise, negative = clockwise

selected_category = 'lens'       
selected_curvature = 'convex'    
selected_shape = 'circular'      
Surfaces: List['Surface'] = []
history: List[List['Surface']] = []
source_pos = np.array([4.0, 10.0])

# Matplotlib Figure Setup
fig = plt.figure(figsize=(13, 8))
# Main Ray Tracing Canvas on the right
ax = fig.add_axes([0.31, 0.08, 0.66, 0.86])

def setup_axes():
    ax.clear()
    ax.set_aspect("equal")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title("Interactive Optical Ray Tracing", fontsize=14, pad=10)

# Optical Ray Tracing Object
class Ray:
    def __init__(self, origin, direction, surfaces, bounces=10, color="orange", in_glass=False, last_surface=None):
        self.origin = np.array(origin, dtype=float)
        norm_dir = np.linalg.norm(direction)
        self.direction = direction / norm_dir if norm_dir > 0 else np.array([1.0, 0.0])
        self.length = 100
        self.surfaces: List['Surface'] = surfaces
        self.bounces = bounces #---Ray Bounce Limit to prevent infinite bouncing
        self.color = color     #---Ray Colour
        self.in_glass = in_glass
        self.last_surface = last_surface

        # Default parametric ray equation
        self.end = self.origin + self.length * self.direction
        # Checking if Ray intersects with any Surface
        self.points = []
        for surface in self.surfaces:
            # Aperture check for each surface
            t_intersection = surface.ray_intersection(self)
            if t_intersection:
                for t_val in t_intersection:
                    # Preventing immediate self-intersection
                    if surface == self.last_surface and t_val < 1e-3:
                        continue
                    self.points.append((t_val, surface))
        # Bounce check
        if self.bounces <= 0:
            if self.points:
                t, surface = min(self.points, key=lambda x: x[0])  # Avoids jumping rays
                self.end = self.origin + t*self.direction
            self.draw()
            return
        else:
            if self.points:
                t, surface = min(self.points, key=lambda x: x[0])   # Finds point of intersection closest to ray, i.e the surface which the ray intersects first, to prevent ghost rays.
                self.end = self.origin + t*self.direction
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
        norm_val = np.linalg.norm(normal)

        if norm_val < 1e-10:
            return
        normal = normal / norm_val

        # Standard Mirror Reflection
        if not surface.is_lens:
            dot = np.dot(incident_dir, normal)
            reflected_dir = incident_dir -2*dot*normal
            reflected_dir /= np.linalg.norm(reflected_dir)

            Ray(
                origin=self.end,
                direction=reflected_dir,
                surfaces=self.surfaces,
                bounces=self.bounces-1,
                color="blue",
                in_glass=self.in_glass,
                last_surface=surface
            )

        # Standard Lens Refraction (Snell's Law)
        else:
            dot = np.dot(incident_dir, normal)
            # Orienting normal as per dot
            oriented_normal = normal if dot < 0 else -normal

            # Setting up Mu
            if not self.in_glass:
                n1, n2 = air_refractive_index, surface.refractive_index
            else:
                n1, n2 = surface.refractive_index, air_refractive_index

            n = n1 / n2
            cos_i = -np.dot(incident_dir, oriented_normal)  # The main dot product
            sin2_t = (n**2) * (1.0 - cos_i**2)              # Used for checking total internal reflection(negative root)

            # Total Internal Reflection Check
            if sin2_t > 1.0:
                refracted_dir = incident_dir -2*(-cos_i)*oriented_normal
                glasscheck = self.in_glass
            else:
                cos_t = math.sqrt(max(0.0, 1.0-sin2_t))
                refracted_dir = n * incident_dir + (n * cos_i - cos_t) * oriented_normal
                glasscheck = not self.in_glass

            refracted_dir /= np.linalg.norm(refracted_dir)

            Ray(
                origin=self.end,
                direction=refracted_dir,
                surfaces=self.surfaces,
                bounces=self.bounces-1,
                color="blue",
                in_glass=glasscheck,
                last_surface=surface
            )

# Coordinate Space Handler Object
class CoordinateSpaceHandler():
    def __init__(self):
        pass

    def localRotate(self, vector: np.ndarray, angle: float):
        rotate = np.array([
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)]
        ])  # Rotate global coordinate system into surface's local system
        return np.dot(rotate, vector)
    
    def globalRotate(self, transformed_vector: np.ndarray, angle: float):
        rotate_transpose = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])  # Rotate surface's local system back to original global coordinate system
        return np.dot(rotate_transpose, transformed_vector)


# Surface Parent Class
class Surface(ABC):
    def __init__(self, centre, hyperparameter, is_concave, angle=0.0, parameter=None, is_lens=False, refractive_index=1.5):
        self.centre = np.array(centre, dtype=float)
        self.is_lens = is_lens
        self.is_concave = is_concave
        self.parameter = parameter if parameter is not None else []
        self.refractive_index = refractive_index if self.is_lens else 1.0
        self.angle = angle
        self.rotationhandler = CoordinateSpaceHandler()

    @abstractmethod
    def normal(self, point: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def ray_intersection(self, ray: Ray) -> List[float]:
        pass

    @abstractmethod
    def surface_intersection(self, surface) -> List[float]:
        pass

    @abstractmethod
    def draw(self):
        pass

# Circular Surface Class
class Circle(Surface):
    def __init__(self, centre, hyperparameter, is_concave, angle=0.0, parameter=None, is_lens=False, refractive_index=1.5):
        if parameter is None and not is_lens:
            parameter = [-45, 45]
        super().__init__(centre, hyperparameter, is_concave, angle, parameter, is_lens, refractive_index)
        self.radius = hyperparameter
        self.centre_angle = 180 if self.is_concave else 0

    def normal(self, point: np.ndarray) -> np.ndarray:
        return np.array([point[0] - self.centre[0], point[1] - self.centre[1]])

    def ray_intersection(self, ray: Ray) -> List[float]:
        p_local = self.rotationhandler.localRotate(ray.origin - self.centre, self.angle)
        d_local = self.rotationhandler.localRotate(ray.direction, self.angle)
        f = p_local[0]
        g = p_local[1]

        Beta = f * d_local[0] + g * d_local[1]
        Delta = f**2 + g**2 - self.radius**2
        discriminant = 4 * (Beta**2) - 4 * Delta

        hits = []
        if discriminant >= 0:
            t1 = (-2 * Beta + discriminant**0.5) / 2
            t2 = (-2 * Beta - discriminant**0.5) / 2
            valid_t = sorted([t for t in (t1, t2) if t > 1e-4])
            for pt in valid_t:
                intersection_local = p_local + pt * d_local
                angle_deg = math.degrees(math.atan2(intersection_local[1], intersection_local[0]))
                diff = (angle_deg - self.centre_angle + 180) % 360 -180
                if self.parameter and min(self.parameter) <= diff <= max(self.parameter):
                    hits.append(pt)
        return hits

    def surface_intersection(self, surface: Surface):
        # Localize relative to self.centre along the lens axis
        c1 = np.array([0.0, 0.0])
        c2 = self.rotationhandler.localRotate(surface.centre - self.centre, self.angle)
        surface_radius = surface.radius
        g = c1[0] - c2[0]
        if abs(g) < 1e-9:
            return
        A = 2*(c1[1] - c2[1])
        B = g**2 + c2[1]**2 - c1[1]**2 + self.radius**2 - surface_radius**2
        alpha = A**2 + 4*(g**2)
        beta = 2*A*B - 8*(g**2)*c1[1]
        gamma = B**2 - 4*(g**2)*(self.radius**2) + 4*(g**2)*(c1[1]**2)
        D = beta**2 - 4*alpha*gamma

        if D >= 0:
            ys = [(-beta + D**0.5)/(2*alpha), (-beta - D**0.5)/(2*alpha)]
            angles = []
            for y in ys:
                x = c1[0] - (A*y+B)/(2*g)
                angle = math.degrees(math.atan2(y-c1[1], x-c1[0]))
                diff = (angle-self.centre_angle+180)%360-180
                angles.append(diff)
            self.parameter = sorted(angles)

    def get_coordinate(self, parameter):
        angle_rad = math.radians(self.centre_angle + parameter)
        local_pt = np.array([
            self.radius * math.cos(angle_rad),
            self.radius * math.sin(angle_rad)
        ])
        return self.centre + self.rotationhandler.globalRotate(local_pt, self.angle)

    def draw(self):
        """Draws the surface on graph as per its mathematical nature."""
        if not self.parameter:
            return
        theta1 = self.centre_angle + min(self.parameter)
        theta2 = self.centre_angle + max(self.parameter)
        thetas = np.linspace(theta1, theta2, 500)
        rads = np.radians(thetas)
        local_curve = np.array([self.radius*np.cos(rads), self.radius*np.sin(rads)])
        global_curve = self.centre[:, None] + self.rotationhandler.globalRotate(local_curve, self.angle)
        points = np.column_stack((global_curve[0], global_curve[1]))
        path = Path(points)
        patch = PathPatch(path, fill=False, linewidth=3, color='black')
        ax.add_patch(patch)

# Parabolic Surface Class
class Parabola(Surface):
    def __init__(self, centre, hyperparameter, is_concave, angle=0.0, parameter=None, is_lens=False, refractive_index=1.5):
        if parameter is None and not is_lens:
            parameter = [-1.5, 1.5]
        super().__init__(centre, hyperparameter, is_concave, angle, parameter, is_lens, refractive_index)
        self.focus = hyperparameter
        self.orientation = -1 if is_concave else 1

    def normal(self, point: np.ndarray) -> np.ndarray:
        p_local = self.rotationhandler.localRotate(point - self.centre, self.angle)
        local_norm = np.array([-4*self.orientation*self.focus, 2*p_local[1]])
        return self.rotationhandler.globalRotate(local_norm, self.angle)

    def ray_intersection(self, ray: Ray) -> List[float]:
        if not self.parameter:
            return []
        p_local = self.rotationhandler.localRotate(ray.origin - self.centre, self.angle)
        d_local = self.rotationhandler.localRotate(ray.direction, self.angle)
        f = p_local[1]
        g = p_local[0]
        A = d_local[1]**2
        B = (2*f*d_local[1] + 4*self.focus*d_local[0]) if self.is_concave else (2*f*d_local[1] - 4*self.focus*d_local[0])
        C = (4*self.focus*g + f**2) if self.is_concave else (f**2 - 4*self.focus*g)
        D = B**2 - 4*A*C

        hits = []
        if abs(A) > 1e-9:
            if D >= 0:
                t1 = (-B + D**0.5) / (2*A)
                t2 = (-B - D**0.5) / (2*A)
                valid_t = sorted([t for t in (t1, t2) if t > 1e-4])
                for pt in valid_t:
                    param = (f + pt*d_local[1]) / (2 * self.orientation * self.focus)
                    if min(self.parameter) <= param <= max(self.parameter):
                        hits.append(pt)
        else:
            if abs(B) > 1e-9:
                pt = -C/B
                if pt > 1e-4:
                    param = (f + pt*d_local[1]) / (2 * self.orientation * self.focus)
                    if min(self.parameter) <= param <= max(self.parameter):
                        hits.append(pt)
        return hits

    def surface_intersection(self, surface: Surface):
        c1 = np.array([0.0, 0.0])
        c2 = self.rotationhandler.localRotate(surface.centre - self.centre, self.angle)
        surface_focus = surface.focus
        s1 = self.orientation
        s2 = surface.orientation
        f1 = self.focus
        f2 = surface_focus
        h1, k1 = c1
        h2, k2 = c2
        A = s2*f2 - s1*f1
        B = 2*(s1*f1*k2 - s2*f2*k1)
        C = (s2*f2*k1**2- s1*f1*k2**2 + 4*s1*s2*f1*f2*(h1-h2))
        discriminant = B**2 - 4*A*C
        if discriminant >= 0:
            D = math.sqrt(discriminant)
            roots = [
                (-B+D)/(2*A),
                (-B-D)/(2*A)
            ]
            self.parameter = []
            for root in roots:
                self.parameter.append(
                    (root-k1)/(2*s1*f1)
                )
            self.parameter = sorted(self.parameter)

    def get_parameter(self, point):
        p_local = self.rotationhandler.localRotate(point - self.centre, self.angle)
        return p_local[1] / (2 * self.orientation * self.focus)

    def get_coordinate(self, parameter):
        local_pt = np.array([
            self.orientation*self.focus*(parameter**2),
            2*self.orientation*self.focus*parameter
        ])
        return self.centre + self.rotationhandler.globalRotate(local_pt, self.angle)

    def draw(self):
        if not self.parameter:
            return
        t = np.linspace(min(self.parameter), max(self.parameter), 500)
        local_curve = np.array([self.orientation*self.focus*t**2, 2*self.orientation*self.focus*t])
        global_curve = self.centre[:, None] + self.rotationhandler.globalRotate(local_curve, self.angle)
        points = np.column_stack((global_curve[0], global_curve[1]))
        path = Path(points)
        patch = PathPatch(path, fill=False, linewidth=3, color='black')
        ax.add_patch(patch)


class Line(Surface):
    def __init__(self, centre, hyperparameter, is_concave=False, angle=0.0, parameter=None, is_lens=False, refractive_index=1.5):
        super().__init__(centre, hyperparameter, is_concave, angle, parameter, is_lens, refractive_index)
        self.origin = self.centre
        direction_vec = np.array(hyperparameter, dtype=float)
        norm = np.linalg.norm(direction_vec)
        self.direction = direction_vec / norm if norm > 0 else np.array([1.0, 0.0])
        self.length = parameter if parameter is not None else 1.0
        self.end = self.origin + self.length*self.direction

    def normal(self, point: np.ndarray = None) -> np.ndarray:
        return np.array([-self.direction[1], self.direction[0]])

    def ray_intersection(self, ray: Ray) -> List[float]:
        p1 = ray.origin
        p2 = self.end
        delX = p2[0] - p1[0]
        delY = p2[1] - p1[1]
        phi = math.atan2(self.direction[1], self.direction[0])
        theita = math.atan2(ray.direction[1], ray.direction[0])
        denom = math.sin(phi - theita)
        if abs(denom) < 1e-10:
            return []
        t = (delX*math.sin(phi) -delY*math.cos(phi))/denom
        if t <= 1e-4:
            return []

        intersection = ray.origin + t*ray.direction
        segmentV = p2 - self.origin
        intersectionV = intersection - self.origin
        projection = np.dot(intersectionV, segmentV)
        seg_len_sq = np.dot(segmentV, segmentV)

        if -1e-5 <= projection <= seg_len_sq + 1e-5:
            return [t]
        return []

    def surface_intersection(self, surface) -> List[float]:
        return []

    def draw(self):
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


class Lens:
    def __init__(self, origin, hp1, hp2, thickness, lenstype, refractive_index=1.5, angle=0.0, aperture=0.0):
        self.origin = np.array(origin, dtype=float)
        self.thickness = thickness
        self.hp1 = hp1
        self.hp2 = hp2
        self.lenstype = lenstype
        self.refractive_index = refractive_index
        self.angle = angle
        self.aperture = aperture
        self.lens_obj = []
        self.rotationhandler = CoordinateSpaceHandler()

    def draw_convex(self):
        axis_dir = np.array([math.cos(self.angle), math.sin(self.angle)])
        if self.lenstype == "circular":
            centre1 = self.origin + (-(self.thickness/2) + self.hp1) * axis_dir
            centre2 = self.origin + ( (self.thickness/2) - self.hp2) * axis_dir
            surface1: Surface = Circle(centre1, self.hp1, is_concave=True, angle=self.angle, is_lens=True, refractive_index=self.refractive_index)
            surface2: Surface = Circle(centre2, self.hp2, is_concave=False, angle=self.angle, is_lens=True, refractive_index=self.refractive_index)
        else:
            centre1 = self.origin - (self.thickness/2) * axis_dir
            centre2 = self.origin + (self.thickness/2) * axis_dir
            surface1: Surface = Parabola(centre1, self.hp1, is_concave=False, angle=self.angle, is_lens=True, refractive_index=self.refractive_index)
            surface2: Surface = Parabola(centre2, self.hp2, is_concave=True, angle=self.angle, is_lens=True, refractive_index=self.refractive_index)

        self.lens_obj.extend([surface1, surface2])
        surface1.surface_intersection(surface2)
        surface2.surface_intersection(surface1)
        self.draw_lens()
        return self.lens_obj

    def draw_concave(self):
        axis_dir = np.array([math.cos(self.angle), math.sin(self.angle)])
        if self.lenstype == "circular":
            c1_concave = self.origin + (-(self.thickness/2) - self.hp1) * axis_dir
            c2_concave = self.origin + ( (self.thickness/2) + self.hp2) * axis_dir

            # If a specific aperture arc is supplied, use it; otherwise auto-intersect like convex
            if self.aperture > 0:
                param = [-self.aperture, self.aperture]
                surface1: Surface = Circle(c1_concave, self.hp1, is_concave=False, angle=self.angle, parameter=param, is_lens=True, refractive_index=self.refractive_index)
                surface2: Surface = Circle(c2_concave, self.hp2, is_concave=True, angle=self.angle, parameter=param, is_lens=True, refractive_index=self.refractive_index)
            else:
                c1_convex = self.origin + (-(self.thickness/2) + self.hp1) * axis_dir
                c2_convex = self.origin + ( (self.thickness/2) - self.hp2) * axis_dir
                surface1: Surface = Circle(c1_convex, self.hp1, is_concave=True, angle=self.angle, is_lens=True, refractive_index=self.refractive_index)
                surface2: Surface = Circle(c2_convex, self.hp2, is_concave=False, angle=self.angle, is_lens=True, refractive_index=self.refractive_index)

                surface1.surface_intersection(surface2)
                surface2.surface_intersection(surface1)

                surface1.centre = c1_concave
                surface2.centre = c2_concave
                surface1.is_concave = False
                surface1.centre_angle = 0
                surface2.is_concave = True
                surface2.centre_angle = 180
        else:
            c1_concave = self.origin - (self.thickness/2) * axis_dir
            c2_concave = self.origin + (self.thickness/2) * axis_dir

            if self.aperture > 0:
                param = [-self.aperture, self.aperture]
                surface1: Surface = Parabola(c1_concave, self.hp1, is_concave=True, angle=self.angle, parameter=param, is_lens=True, refractive_index=self.refractive_index)
                surface2: Surface = Parabola(c2_concave, self.hp2, is_concave=False, angle=self.angle, parameter=param, is_lens=True, refractive_index=self.refractive_index)
            else:
                c1_convex = self.origin - (self.thickness/2) * axis_dir
                c2_convex = self.origin + (self.thickness/2) * axis_dir
                surface1: Surface = Parabola(c1_convex, self.hp1, is_concave=False, angle=self.angle, is_lens=True, refractive_index=self.refractive_index)
                surface2: Surface = Parabola(c2_convex, self.hp2, is_concave=True, angle=self.angle, is_lens=True, refractive_index=self.refractive_index)

                surface1.surface_intersection(surface2)
                surface2.surface_intersection(surface1)

                surface1.centre = c1_concave
                surface2.centre = c2_concave
                surface1.is_concave = True
                surface1.orientation = -1
                surface2.is_concave = False
                surface2.orientation = 1

        self.lens_obj.extend([surface1, surface2])

        perp_dir = np.array([-math.sin(self.angle), math.cos(self.angle)])
        endpoints1 = sorted(
            [surface1.get_coordinate(surface1.parameter[0]), surface1.get_coordinate(surface1.parameter[1])],
            key=lambda p: np.dot(p, perp_dir)
        )
        endpoints2 = sorted(
            [surface2.get_coordinate(surface2.parameter[0]), surface2.get_coordinate(surface2.parameter[1])],
            key=lambda p: np.dot(p, perp_dir)
        )

        dir_vec = np.array([math.cos(self.angle), math.sin(self.angle)])
        len_top = np.linalg.norm(endpoints2[1]-endpoints1[1])
        len_bot = np.linalg.norm(endpoints2[0]-endpoints1[0])

        surface3 = Line(endpoints1[1], dir_vec, is_concave=False, angle=self.angle, parameter=len_top, is_lens=True, refractive_index=self.refractive_index)
        surface4 = Line(endpoints1[0], dir_vec, is_concave=False, angle=self.angle, parameter=len_bot, is_lens=True, refractive_index=self.refractive_index)

        self.lens_obj.extend([surface3, surface4])
        self.draw_lens()
        return self.lens_obj

    def draw_lens(self):
        for surface in self.lens_obj:
            surface.draw()


class Source:
    def __init__(self, origin, surfaces: List[Surface], rays=32):
        self.rays = []
        self.origin = np.array(origin, dtype=float)
        self.surfaces = surfaces
        n = 360/rays

        for i in range(rays):
            a = np.radians(i*n)
            d = np.array([np.cos(a), np.sin(a)])
            self.rays.append(Ray(self.origin, d, self.surfaces))

        ax.scatter(self.origin[0], self.origin[1], color="red", zorder=5, label="Source")


# UI, Controls, Layout & Styling 
ACTIVE_BG = '#a8e6cf'    # Pastel Green for active choice
INACTIVE_BG = '#f0f0f0'  # Clean light gray for inactive

# Sub-header text elements
fig.text(0.03, 0.965, "OBJECT TYPE", fontsize=8.5, fontweight='bold', color='#444')
fig.text(0.03, 0.850, "CURVATURE", fontsize=8.5, fontweight='bold', color='#444')
fig.text(0.03, 0.775, "GEOMETRY", fontsize=8.5, fontweight='bold', color='#444')
fig.text(0.03, 0.700, "PARAMETERS", fontsize=8.5, fontweight='bold', color='#444')

## UI Elements:
# 1. Object Type Buttons
ax_btn_source = fig.add_axes([0.03, 0.925, 0.24, 0.034])
ax_btn_mirror = fig.add_axes([0.03, 0.885, 0.115, 0.034])
ax_btn_lens   = fig.add_axes([0.155, 0.885, 0.115, 0.034])

btn_source = Button(ax_btn_source, "Move Source (s)")
btn_mirror = Button(ax_btn_mirror, "Mirror (m)")
btn_lens   = Button(ax_btn_lens, "Lens (l)")

# 2. Curvature Buttons
ax_btn_convex  = fig.add_axes([0.03, 0.810, 0.115, 0.034])
ax_btn_concave = fig.add_axes([0.155, 0.810, 0.115, 0.034])

btn_convex  = Button(ax_btn_convex, "Convex (v)")
btn_concave = Button(ax_btn_concave, "Concave (n)")

# 3. Geometry Buttons
ax_btn_circular  = fig.add_axes([0.03, 0.735, 0.115, 0.034])
ax_btn_parabolic = fig.add_axes([0.155, 0.735, 0.115, 0.034])

btn_circular  = Button(ax_btn_circular, "Circular (r)")
btn_parabolic = Button(ax_btn_parabolic, "Parabolic (p)")

# 4. TextBoxes (Parameters)
ax_tb_param    = fig.add_axes([0.135, 0.658, 0.135, 0.028])
ax_tb_param2   = fig.add_axes([0.135, 0.622, 0.135, 0.028])
ax_tb_thick    = fig.add_axes([0.135, 0.586, 0.135, 0.028])
ax_tb_aperture = fig.add_axes([0.135, 0.550, 0.135, 0.028])
ax_tb_angle    = fig.add_axes([0.135, 0.514, 0.135, 0.028])
ax_tb_air      = fig.add_axes([0.135, 0.478, 0.135, 0.028])
ax_tb_glass    = fig.add_axes([0.135, 0.442, 0.135, 0.028])

tb_param    = TextBox(ax_tb_param, "Radius 1: ", initial=str(hyperparameter_val))
tb_param2   = TextBox(ax_tb_param2, "Radius 2: ", initial=str(hyperparameter_val2))
tb_thick    = TextBox(ax_tb_thick, "Thickness: ", initial=str(thickness_val))
tb_aperture = TextBox(ax_tb_aperture, "Aperture: ", initial=str(concave_aperture_val))
tb_angle    = TextBox(ax_tb_angle, "Angle (°): ", initial=str(angle_val))
tb_air      = TextBox(ax_tb_air, "Air (n): ", initial=str(air_refractive_index))
tb_glass    = TextBox(ax_tb_glass, "Glass (n): ", initial=str(glass_refractive_index))

# Angle Orientation Guide
angle_info_text = fig.text(
    0.03, 0.410,
    "Angle: (+) anti-clockwise, (-) clockwise",
    fontsize=7.8, style='italic', color='#555'
)

# 5. Action Buttons (Undo & Clear)
ax_btn_undo  = fig.add_axes([0.03, 0.355, 0.24, 0.034])
ax_btn_clear = fig.add_axes([0.03, 0.315, 0.24, 0.034])

btn_undo  = Button(ax_btn_undo, "Undo (c)", color='#fff0f0', hovercolor='#ffd8d8')
btn_clear = Button(ax_btn_clear, "Clear All (x)", color='#fff0f0', hovercolor='#ffd8d8')

# Ensure clean font sizing across buttons
for b in [btn_source, btn_mirror, btn_lens, btn_convex, btn_concave, btn_circular, btn_parabolic, btn_undo, btn_clear]:
    b.label.set_fontsize(8.5)

status_box = fig.text(
    0.03, 0.06, "",
    fontsize=8.5, style='italic', color='#222',
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#f9f9f9", edgecolor="#ddd")
)

# Widget reference list
widgets = [
    btn_source, btn_mirror, btn_lens,
    btn_convex, btn_concave,
    btn_circular, btn_parabolic,
    tb_param, tb_param2, tb_thick, tb_aperture, tb_angle, tb_air, tb_glass,
    btn_undo, btn_clear
]

def update_ui_state():
    """Updates button highlight colors and dynamic labels according to the selected state."""
    # Category buttons
    btn_source.color = ACTIVE_BG if selected_category == 'source' else INACTIVE_BG
    btn_mirror.color = ACTIVE_BG if selected_category == 'mirror' else INACTIVE_BG
    btn_lens.color   = ACTIVE_BG if selected_category == 'lens' else INACTIVE_BG
    btn_convex.color  = ACTIVE_BG if selected_curvature == 'convex' else INACTIVE_BG
    btn_concave.color = ACTIVE_BG if selected_curvature == 'concave' else INACTIVE_BG
    btn_circular.color  = ACTIVE_BG if selected_shape == 'circular' else INACTIVE_BG
    btn_parabolic.color = ACTIVE_BG if selected_shape == 'parabolic' else INACTIVE_BG

    is_lens_mode = (selected_category == 'lens')
    is_concave_lens = (is_lens_mode and selected_curvature == 'concave')
    is_source_mode = (selected_category == 'source')

    # Dynamic visibility of parameters
    ax_tb_param2.set_visible(is_lens_mode)
    ax_tb_thick.set_visible(is_lens_mode)
    ax_tb_aperture.set_visible(is_concave_lens)
    ax_tb_angle.set_visible(not is_source_mode)

    if is_lens_mode:
        if selected_shape == 'circular':
            tb_param.label.set_text("Radius 1: ")
            tb_param2.label.set_text("Radius 2: ")
            tb_aperture.label.set_text("Arc (°): ")
        else:
            tb_param.label.set_text("Focus 1: ")
            tb_param2.label.set_text("Focus 2: ")
            tb_aperture.label.set_text("Aperture: ")
    else:
        if selected_shape == 'circular':
            tb_param.label.set_text("Radius: ")
        else:
            tb_param.label.set_text("Focus: ")

    # Update Status Box message
    angle_note = "Angle: (+) anti-clockwise, (-) clockwise."
    if selected_category == 'source':
        status_box.set_text(
            "Action:\nClick on the grid to move\nthe light source.\n\n"
            "Hotkeys: s, m, l, v, n, r, p, c, x"
        )
    elif selected_category == 'lens':
        label = "Radius" if selected_shape == 'circular' else "Focus"
        if selected_curvature == 'concave':
            ap_str = f", Arc = {concave_aperture_val}°" if (selected_shape == 'circular' and concave_aperture_val > 0) else (f", Ap = {concave_aperture_val}" if concave_aperture_val > 0 else ", Ap = Auto")
            status_box.set_text(
                f"Action:\nClick grid to place Concave\n"
                f"{selected_shape.capitalize()} Lens.\n"
                f"({label} 1 = {hyperparameter_val}, 2 = {hyperparameter_val2},\n"
                f"Thick = {thickness_val}{ap_str}, Angle = {angle_val}°)\n"
                f"{angle_note}"
            )
        else:
            status_box.set_text(
                f"Action:\nClick grid to place Convex\n"
                f"{selected_shape.capitalize()} Lens.\n"
                f"({label} 1 = {hyperparameter_val}, 2 = {hyperparameter_val2},\n"
                f"Thick = {thickness_val}, Angle = {angle_val}°)\n"
                f"{angle_note}"
            )
    else:
        label = "Radius" if selected_shape == 'circular' else "Focus"
        status_box.set_text(
            f"Action:\nClick grid to place {selected_curvature.capitalize()}\n"
            f"{selected_shape.capitalize()} Mirror.\n"
            f"({label} = {hyperparameter_val}, Angle = {angle_val}°)\n"
            f"{angle_note}"
        )

    fig.canvas.draw_idle()

def redraw():
    setup_axes()
    for s in Surfaces:
        s.draw()
    Source(source_pos, Surfaces)
    fig.canvas.draw_idle()

# Callback Handlers
def set_category(cat):
    global selected_category
    selected_category = cat
    update_ui_state()

def set_curvature(curv):
    global selected_curvature
    selected_curvature = curv
    update_ui_state()

def set_shape(shp):
    global selected_shape
    selected_shape = shp
    update_ui_state()

btn_source.on_clicked(lambda event: set_category('source'))
btn_mirror.on_clicked(lambda event: set_category('mirror'))
btn_lens.on_clicked(lambda event: set_category('lens'))

btn_convex.on_clicked(lambda event: set_curvature('convex'))
btn_concave.on_clicked(lambda event: set_curvature('concave'))

btn_circular.on_clicked(lambda event: set_shape('circular'))
btn_parabolic.on_clicked(lambda event: set_shape('parabolic'))

def on_param_submit(text):
    global hyperparameter_val
    try:
        val = float(text)
        if val > 0:
            hyperparameter_val = val
            update_ui_state()
    except ValueError:
        pass

def on_param2_submit(text):
    global hyperparameter_val2
    try:
        val = float(text)
        if val > 0:
            hyperparameter_val2 = val
            update_ui_state()
    except ValueError:
        pass

def on_thick_submit(text):
    global thickness_val
    try:
        val = float(text)
        if val > 0:
            thickness_val = val
            update_ui_state()
    except ValueError:
        pass

def on_aperture_submit(text):
    global concave_aperture_val
    try:
        val = float(text)
        if val >= 0:
            concave_aperture_val = val
            update_ui_state()
    except ValueError:
        pass

def on_angle_submit(text):
    global angle_val
    try:
        val = float(text)
        angle_val = val
        update_ui_state()
    except ValueError:
        pass

def on_air_submit(text):
    global air_refractive_index
    try:
        val = float(text)
        if val > 0:
            air_refractive_index = val
            redraw()
    except ValueError:
        pass

def on_glass_submit(text):
    global glass_refractive_index
    try:
        val = float(text)
        if val > 0:
            glass_refractive_index = val
            for s in Surfaces:
                if s.is_lens:
                    s.refractive_index = glass_refractive_index
            redraw()
    except ValueError:
        pass

tb_param.on_submit(on_param_submit)
tb_param2.on_submit(on_param2_submit)
tb_thick.on_submit(on_thick_submit)
tb_aperture.on_submit(on_aperture_submit)
tb_angle.on_submit(on_angle_submit)
tb_air.on_submit(on_air_submit)
tb_glass.on_submit(on_glass_submit)

def on_undo(event):
    if history:
        last_group = history.pop()
        for item in last_group:
            if item in Surfaces:
                Surfaces.remove(item)
        redraw()

def on_clear(event):
    Surfaces.clear()
    history.clear()
    redraw()

btn_undo.on_clicked(on_undo)
btn_clear.on_clicked(on_clear)

# Keyboard Hotkeys Listener
def on_key_press(event):
    # Prevent hotkeys while user is typing into any TextBox
    for w in [tb_param, tb_param2, tb_thick, tb_aperture, tb_angle, tb_air, tb_glass]:
        if getattr(w, 'active', False):
            return
    if event.key is None:
        return
    k = event.key.lower()
    if k == 's':
        set_category('source')
    elif k == 'm':
        set_category('mirror')
    elif k == 'l':
        set_category('lens')
    elif k == 'v':
        set_curvature('convex')
    elif k == 'n':
        set_curvature('concave')
    elif k == 'r':
        set_shape('circular')
    elif k == 'p':
        set_shape('parabolic')
    elif k == 'c':
        on_undo(None)
    elif k == 'x':
        on_clear(None)

fig.canvas.mpl_connect('key_press_event', on_key_press)

# Canvas Click Listener
def on_canvas_click(event):
    global source_pos
    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        return

    pos = np.array([event.xdata, event.ydata])
    angle_rad = math.radians(angle_val)

    if selected_category == 'source':
        source_pos = pos
        redraw()

    elif selected_category == 'mirror':
        is_concave = (selected_curvature == 'concave')
        if selected_shape == 'circular':
            mirror = Circle(pos, hyperparameter=hyperparameter_val, is_concave=is_concave, angle=angle_rad, parameter=[-45, 45])
        else:
            mirror = Parabola(pos, hyperparameter=hyperparameter_val, is_concave=is_concave, angle=angle_rad, parameter=[-1.5, 1.5])
        Surfaces.append(mirror)
        history.append([mirror])
        redraw()

    elif selected_category == 'lens':
        lens = Lens(
            pos,
            hp1=hyperparameter_val,
            hp2=hyperparameter_val2,
            thickness=thickness_val,
            lenstype=selected_shape,
            refractive_index=glass_refractive_index,
            angle=angle_rad,
            aperture=concave_aperture_val
        )
        if selected_curvature == 'convex':
            new_surfs = lens.draw_convex()
        else:
            new_surfs = lens.draw_concave()

        Surfaces.extend(new_surfs)
        history.append(new_surfs)
        redraw()

fig.canvas.mpl_connect('button_press_event', on_canvas_click)

# Initial setup and launch
update_ui_state()
redraw()
plt.show()