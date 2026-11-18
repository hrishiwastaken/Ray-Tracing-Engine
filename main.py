import tkinter as tk
from tkinter import ttk, messagebox, filedialog, colorchooser
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import math
import os
import sys
import json
from abc import ABC, abstractmethod
from typing import List

# ==============================================================================
# Global Physics & App State
# ==============================================================================
air_refractive_index = 1.0
glass_refractive_index = 1.5
hyperparameter_val = 3.0 
hyperparameter_val2 = 3.0
thickness_val = 1.5
mirror_arc_val = 0.0        # 0.0 = Default arc/span (45° circular, 1.5 parabolic)
angle_val = 0.0             # In degrees: positive = anti-clockwise, negative = clockwise

# Viewport / Navigation State
MIN_SPAN = 20.0             # 20x20 minimum zoom
MAX_SPAN = 100.0            # 100x100 maximum zoom
view_xlim = [0.0, 20.0]
view_ylim = [0.0, 20.0]
pan_active = False
pan_start_data = None

# Source Configuration
source_type = 'point'       # 'point' or 'parallel'
source_rays_val = 32
source_span_val = 4.0
source_angle_val = 0.0
source_pos = np.array([4.0, 10.0])

# Styling & Colors
color_source = "#e74c3c"    # Red
color_incident = "#f39c12"  # Orange
color_reflected = "#3498db" # Blue
color_refracted = "#2ecc71" # Green
color_mirror = "#555555"    # Neutral Mirror
color_lens = "#1abc9c"      # Cyan/Teal Lens

selected_category = 'lens'       
selected_curvature = 'convex'    
selected_shape = 'circular'      
Surfaces: List['Surface'] = []
history: List[List['Surface']] = []

# Theme State
is_dark_mode = False

# File Document Management
current_file_path = None
is_modified = False

# ==============================================================================
# Optical Ray Tracing Mathematics & Classes
# ==============================================================================
class Ray:
    def __init__(self, origin, direction, surfaces, bounces=10, color="orange", in_glass=False, last_surface=None,
                 col_reflected="blue", col_refracted="green"):
        self.origin = np.array(origin, dtype=float)
        norm_dir = np.linalg.norm(direction)
        self.direction = direction / norm_dir if norm_dir > 0 else np.array([1.0, 0.0])
        self.length = 300   # Extended length to accommodate 100x100 zoom scale
        self.surfaces: List['Surface'] = surfaces
        self.bounces = bounces
        self.color = color
        self.in_glass = in_glass
        self.last_surface = last_surface
        self.col_reflected = col_reflected
        self.col_refracted = col_refracted

        self.end = self.origin + self.length * self.direction
        self.points = []
        for surface in self.surfaces:
            t_intersection = surface.ray_intersection(self)
            if t_intersection:
                for t_val in t_intersection:
                    if surface == self.last_surface and t_val < 1e-3:
                        continue
                    self.points.append((t_val, surface))

        if self.bounces <= 0:
            if self.points:
                t, surface = min(self.points, key=lambda x: x[0])
                self.end = self.origin + t*self.direction
            self.draw()
            return
        else:
            if self.points:
                t, surface = min(self.points, key=lambda x: x[0])
                self.end = self.origin + t*self.direction
                self.draw()
                self.draw_reflection_or_refraction(surface)
            else:
                self.draw()
            return

    def draw(self):
        z = 2 if self.color != color_incident else 1
        l = 2 if self.color != color_incident else 1.5
        ax.plot(
            [self.origin[0], self.end[0]],
            [self.origin[1], self.end[1]],
            color=self.color,
            alpha=0.9,
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
                color=self.col_reflected,
                in_glass=self.in_glass,
                last_surface=surface,
                col_reflected=self.col_reflected,
                col_refracted=self.col_refracted
            )

        # Standard Lens Refraction (Snell's Law)
        else:
            dot = np.dot(incident_dir, normal)
            oriented_normal = normal if dot < 0 else -normal

            if not self.in_glass:
                n1, n2 = air_refractive_index, surface.refractive_index
            else:
                n1, n2 = surface.refractive_index, air_refractive_index

            n = n1 / n2
            cos_i = -np.dot(incident_dir, oriented_normal)
            sin2_t = (n**2) * (1.0 - cos_i**2)

            if sin2_t > 1.0:
                refracted_dir = incident_dir -2*(-cos_i)*oriented_normal
                glasscheck = self.in_glass
                next_color = self.col_reflected
            else:
                cos_t = math.sqrt(max(0.0, 1.0-sin2_t))
                refracted_dir = n * incident_dir + (n * cos_i - cos_t) * oriented_normal
                glasscheck = not self.in_glass
                next_color = self.col_refracted

            refracted_dir /= np.linalg.norm(refracted_dir)

            Ray(
                origin=self.end,
                direction=refracted_dir,
                surfaces=self.surfaces,
                bounces=self.bounces-1,
                color=next_color,
                in_glass=glasscheck,
                last_surface=surface,
                col_reflected=self.col_reflected,
                col_refracted=self.col_refracted
            )

class CoordinateSpaceHandler:
    def localRotate(self, vector: np.ndarray, angle: float):
        rotate = np.array([
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)]
        ])
        return np.dot(rotate, vector)
    
    def globalRotate(self, transformed_vector: np.ndarray, angle: float):
        rotate_transpose = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        return np.dot(rotate_transpose, transformed_vector)

class Surface(ABC):
    def __init__(self, centre, hyperparameter, is_concave, angle=0.0, parameter=None, is_lens=False, refractive_index=1.5, color=None):
        self.centre = np.array(centre, dtype=float)
        self.is_lens = is_lens
        self.is_concave = is_concave
        self.parameter = parameter if parameter is not None else []
        self.refractive_index = refractive_index if self.is_lens else 1.0
        self.angle = angle
        self.color = color if color else (color_lens if self.is_lens else color_mirror)
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

    @abstractmethod
    def to_dict(self) -> dict:
        pass

class Circle(Surface):
    def __init__(self, centre, hyperparameter, is_concave, angle=0.0, parameter=None, is_lens=False, refractive_index=1.5, color=None):
        if parameter is None and not is_lens:
            parameter = [-45, 45]
        super().__init__(centre, hyperparameter, is_concave, angle, parameter, is_lens, refractive_index, color)
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
        patch = PathPatch(path, fill=False, linewidth=3, color=self.color)
        ax.add_patch(patch)

    def to_dict(self) -> dict:
        return {
            "type": "Circle",
            "centre": self.centre.tolist(),
            "hyperparameter": self.radius,
            "is_concave": self.is_concave,
            "angle": self.angle,
            "parameter": self.parameter,
            "is_lens": self.is_lens,
            "refractive_index": self.refractive_index,
            "color": self.color
        }

class Parabola(Surface):
    def __init__(self, centre, hyperparameter, is_concave, angle=0.0, parameter=None, is_lens=False, refractive_index=1.5, color=None):
        if parameter is None and not is_lens:
            parameter = [-1.5, 1.5]
        super().__init__(centre, hyperparameter, is_concave, angle, parameter, is_lens, refractive_index, color)
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
        patch = PathPatch(path, fill=False, linewidth=3, color=self.color)
        ax.add_patch(patch)

    def to_dict(self) -> dict:
        return {
            "type": "Parabola",
            "centre": self.centre.tolist(),
            "hyperparameter": self.focus,
            "is_concave": self.is_concave,
            "angle": self.angle,
            "parameter": self.parameter,
            "is_lens": self.is_lens,
            "refractive_index": self.refractive_index,
            "color": self.color
        }

class Line(Surface):
    def __init__(self, centre, hyperparameter, is_concave=False, angle=0.0, parameter=None, is_lens=False, refractive_index=1.5, color=None):
        super().__init__(centre, hyperparameter, is_concave, angle, parameter, is_lens, refractive_index, color)
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
        ax.plot(
            [self.origin[0], self.end[0]],
            [self.origin[1], self.end[1]],
            color=self.color,
            alpha=0.9,
            linewidth=3,
            zorder=3
        )

    def to_dict(self) -> dict:
        return {
            "type": "Line",
            "centre": self.centre.tolist(),
            "hyperparameter": self.direction.tolist(),
            "is_concave": self.is_concave,
            "angle": self.angle,
            "parameter": self.length,
            "is_lens": self.is_lens,
            "refractive_index": self.refractive_index,
            "color": self.color
        }

def surface_from_dict(d: dict) -> Surface:
    stype = d["type"]
    if stype == "Circle":
        return Circle(d["centre"], d["hyperparameter"], d["is_concave"], d["angle"], d["parameter"], d["is_lens"], d["refractive_index"], d.get("color"))
    elif stype == "Parabola":
        return Parabola(d["centre"], d["hyperparameter"], d["is_concave"], d["angle"], d["parameter"], d["is_lens"], d["refractive_index"], d.get("color"))
    elif stype == "Line":
        return Line(d["centre"], d["hyperparameter"], d["is_concave"], d["angle"], d["parameter"], d["is_lens"], d["refractive_index"], d.get("color"))
    raise ValueError(f"Unknown surface type: {stype}")

class Lens:
    def __init__(self, origin, hp1, hp2, thickness, lenstype, refractive_index=1.5, angle=0.0, color=None):
        self.origin = np.array(origin, dtype=float)
        self.thickness = thickness
        self.hp1 = hp1
        self.hp2 = hp2
        self.lenstype = lenstype
        self.refractive_index = refractive_index
        self.angle = angle
        self.color = color if color else color_lens
        self.lens_obj = []
        self.rotationhandler = CoordinateSpaceHandler()

    def draw_convex(self):
        axis_dir = np.array([math.cos(self.angle), math.sin(self.angle)])
        if self.lenstype == "circular":
            centre1 = self.origin + (-(self.thickness/2) + self.hp1) * axis_dir
            centre2 = self.origin + ( (self.thickness/2) - self.hp2) * axis_dir
            surface1: Surface = Circle(centre1, self.hp1, is_concave=True, angle=self.angle, is_lens=True, refractive_index=self.refractive_index, color=self.color)
            surface2: Surface = Circle(centre2, self.hp2, is_concave=False, angle=self.angle, is_lens=True, refractive_index=self.refractive_index, color=self.color)
        else:
            centre1 = self.origin - (self.thickness/2) * axis_dir
            centre2 = self.origin + (self.thickness/2) * axis_dir
            surface1: Surface = Parabola(centre1, self.hp1, is_concave=False, angle=self.angle, is_lens=True, refractive_index=self.refractive_index, color=self.color)
            surface2: Surface = Parabola(centre2, self.hp2, is_concave=True, angle=self.angle, is_lens=True, refractive_index=self.refractive_index, color=self.color)

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

            c1_convex = self.origin + (-(self.thickness/2) + self.hp1) * axis_dir
            c2_convex = self.origin + ( (self.thickness/2) - self.hp2) * axis_dir
            surface1: Surface = Circle(c1_convex, self.hp1, is_concave=True, angle=self.angle, is_lens=True, refractive_index=self.refractive_index, color=self.color)
            surface2: Surface = Circle(c2_convex, self.hp2, is_concave=False, angle=self.angle, is_lens=True, refractive_index=self.refractive_index, color=self.color)

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

            c1_convex = self.origin - (self.thickness/2) * axis_dir
            c2_convex = self.origin + (self.thickness/2) * axis_dir
            surface1: Surface = Parabola(c1_convex, self.hp1, is_concave=False, angle=self.angle, is_lens=True, refractive_index=self.refractive_index, color=self.color)
            surface2: Surface = Parabola(c2_convex, self.hp2, is_concave=True, angle=self.angle, is_lens=True, refractive_index=self.refractive_index, color=self.color)

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

        surface3 = Line(endpoints1[1], dir_vec, is_concave=False, angle=self.angle, parameter=len_top, is_lens=True, refractive_index=self.refractive_index, color=self.color)
        surface4 = Line(endpoints1[0], dir_vec, is_concave=False, angle=self.angle, parameter=len_bot, is_lens=True, refractive_index=self.refractive_index, color=self.color)

        self.lens_obj.extend([surface3, surface4])
        self.draw_lens()
        return self.lens_obj

    def draw_slab(self):
        axis_dir = np.array([math.cos(self.angle), math.sin(self.angle)])
        trans_dir = np.array([-math.sin(self.angle), math.cos(self.angle)])

        w = self.hp1  # Face Width
        t = self.thickness

        p_front_top = self.origin - (t/2)*axis_dir + (w/2)*trans_dir
        p_front_bot = self.origin - (t/2)*axis_dir - (w/2)*trans_dir
        p_back_top  = self.origin + (t/2)*axis_dir + (w/2)*trans_dir
        p_back_bot  = self.origin + (t/2)*axis_dir - (w/2)*trans_dir

        front_face = Line(p_front_bot, trans_dir, is_concave=False, angle=self.angle, parameter=w, is_lens=True, refractive_index=self.refractive_index, color=self.color)
        back_face  = Line(p_back_bot,  trans_dir, is_concave=False, angle=self.angle, parameter=w, is_lens=True, refractive_index=self.refractive_index, color=self.color)
        top_face   = Line(p_front_top, axis_dir,  is_concave=False, angle=self.angle, parameter=t, is_lens=True, refractive_index=self.refractive_index, color=self.color)
        bot_face   = Line(p_front_bot, axis_dir,  is_concave=False, angle=self.angle, parameter=t, is_lens=True, refractive_index=self.refractive_index, color=self.color)

        self.lens_obj = [front_face, back_face, top_face, bot_face]
        self.draw_lens()
        return self.lens_obj

    def draw_lens(self):
        for surface in self.lens_obj:
            surface.draw()

class Source:
    def __init__(self, origin, surfaces: List[Surface], s_type='point', rays=32, span=4.0, angle=0.0):
        self.origin = np.array(origin, dtype=float)
        self.surfaces = surfaces
        self.s_type = s_type
        self.rays_count = max(1, int(rays))
        self.span = float(span)
        self.angle = float(angle)
        self.rays = []

        if self.s_type == 'parallel':
            rad = math.radians(self.angle)
            d = np.array([math.cos(rad), math.sin(rad)])
            n_span = np.array([-math.sin(rad), math.cos(rad)])

            if self.rays_count == 1:
                offsets = [0.0]
            else:
                offsets = np.linspace(-self.span/2, self.span/2, self.rays_count)

            for offset in offsets:
                r_orig = self.origin + offset * n_span
                self.rays.append(Ray(
                    r_orig, d, self.surfaces,
                    color=color_incident,
                    col_reflected=color_reflected,
                    col_refracted=color_refracted
                ))

            p1 = self.origin - (self.span/2) * n_span
            p2 = self.origin + (self.span/2) * n_span
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color_source, linewidth=3.5, zorder=6)
            ax.scatter(self.origin[0], self.origin[1], color=color_source, s=25, zorder=7)
        else:
            step = 360.0 / self.rays_count
            for i in range(self.rays_count):
                a = np.radians(i * step)
                d = np.array([np.cos(a), np.sin(a)])
                self.rays.append(Ray(
                    self.origin, d, self.surfaces,
                    color=color_incident,
                    col_reflected=color_reflected,
                    col_refracted=color_refracted
                ))
            ax.scatter(self.origin[0], self.origin[1], color=color_source, s=40, zorder=7)


# ==============================================================================
# Tkinter Window & Matplotlib Canvas Integration
# ==============================================================================
root = tk.Tk()
root.geometry("1320x860")
root.minsize(1050, 720)

def update_window_title():
    doc_name = os.path.basename(current_file_path) if current_file_path else "Untitled"
    marker = "*" if is_modified else ""
    root.title(f"{marker}{doc_name} - Optical Workbench")

def set_modified(val=True):
    global is_modified
    if is_modified != val:
        is_modified = val
        update_window_title()

# Master Layout
sidebar_container = tk.Frame(root, width=340, relief=tk.RIDGE, bd=1)
sidebar_container.pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=4)
sidebar_container.pack_propagate(False)

plot_frame = tk.Frame(root)
plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=4, pady=4)

fig, ax = plt.subplots(figsize=(8, 8))
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(fill=tk.BOTH, expand=True)

def setup_axes():
    ax.clear()
    ax.set_aspect("equal")
    ax.set_xlim(view_xlim)
    ax.set_ylim(view_ylim)
    
    if is_dark_mode:
        fig.patch.set_facecolor('#121212')
        ax.set_facecolor('#181818')
        ax.tick_params(colors='#aaaaaa')
        for spine in ax.spines.values():
            spine.set_color('#444444')
        ax.grid(True, linestyle='--', alpha=0.3, color='#888888')
        ax.set_title("Optical Workbench Simulation", fontsize=12, pad=8, color='#ffffff')
    else:
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#ffffff')
        ax.tick_params(colors='#333333')
        for spine in ax.spines.values():
            spine.set_color('#cccccc')
        ax.grid(True, linestyle='--', alpha=0.5, color='#999999')
        ax.set_title("Optical Workbench Simulation", fontsize=12, pad=8, color='#000000')

def redraw():
    setup_axes()
    for s in Surfaces:
        s.draw()
    Source(source_pos, Surfaces, s_type=source_type, rays=source_rays_val, span=source_span_val, angle=source_angle_val)
    canvas.draw_idle()

def reset_view():
    global view_xlim, view_ylim
    view_xlim = [0.0, 20.0]
    view_ylim = [0.0, 20.0]
    ax.set_xlim(view_xlim)
    ax.set_ylim(view_ylim)
    canvas.draw_idle()

# Top Header Bar in Sidebar
header_frame = tk.Frame(sidebar_container)
header_frame.pack(fill=tk.X, padx=4, pady=(2, 4))

lbl_app_title = tk.Label(header_frame, text="OPTICAL WORKBENCH", font=("Helvetica", 9, "bold"))
lbl_app_title.pack(side=tk.LEFT, anchor="w")

btn_reset_view = tk.Button(header_frame, text="Reset View", font=("Helvetica", 7, "bold"), relief=tk.GROOVE, command=reset_view)
btn_reset_view.pack(side=tk.RIGHT, padx=(2, 0))

btn_theme = tk.Button(header_frame, text="🌙 Dark Mode", font=("Helvetica", 7, "bold"), relief=tk.GROOVE)
btn_theme.pack(side=tk.RIGHT)

# ==============================================================================
# Tabbed Interface Layout (ttk.Notebook)
# ==============================================================================
notebook = ttk.Notebook(sidebar_container)
notebook.pack(fill=tk.BOTH, expand=True)

tab_optics = tk.Frame(notebook, padx=8, pady=8)
tab_source = tk.Frame(notebook, padx=8, pady=8)
tab_colors = tk.Frame(notebook, padx=8, pady=8)
tab_file   = tk.Frame(notebook, padx=8, pady=8)

notebook.add(tab_optics, text="⚡ Optics")
notebook.add(tab_source, text="💡 Source")
notebook.add(tab_colors, text="🎨 Colors")
notebook.add(tab_file,   text="💾 Workbench")

all_entry_widgets = []
all_label_widgets = []

def make_entry_row(parent, label_text, default_val):
    row = tk.Frame(parent)
    lbl = tk.Label(row, text=label_text, font=("Helvetica", 8), width=13, anchor="w")
    lbl.pack(side=tk.LEFT)
    all_label_widgets.append(lbl)

    ent = tk.Entry(row, font=("Helvetica", 8), relief=tk.FLAT, bd=1, highlightthickness=1)
    ent.insert(0, str(default_val))
    ent.pack(side=tk.RIGHT, fill=tk.X, expand=True)
    all_entry_widgets.append(ent)
    return row, lbl, ent

# ------------------------------------------------------------------------------
# TAB 1: Optics Controls
# ------------------------------------------------------------------------------
lbl_type = tk.Label(tab_optics, text="OBJECT TYPE", font=("Helvetica", 8, "bold"))
lbl_type.pack(anchor="w")
all_label_widgets.append(lbl_type)

frame_obj = tk.Frame(tab_optics)
frame_obj.pack(fill=tk.X, pady=(2, 6))

btn_source = tk.Button(frame_obj, text="Move Source (s)", font=("Helvetica", 8), relief=tk.GROOVE)
btn_source.pack(fill=tk.X, pady=(0, 2))

sub_obj = tk.Frame(frame_obj)
sub_obj.pack(fill=tk.X)
btn_mirror = tk.Button(sub_obj, text="Mirror (m)", font=("Helvetica", 8), relief=tk.GROOVE)
btn_mirror.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
btn_lens = tk.Button(sub_obj, text="Lens (l)", font=("Helvetica", 8), relief=tk.GROOVE)
btn_lens.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))

lbl_curv = tk.Label(tab_optics, text="CURVATURE", font=("Helvetica", 8, "bold"))
lbl_curv.pack(anchor="w")
all_label_widgets.append(lbl_curv)

frame_curv = tk.Frame(tab_optics)
frame_curv.pack(fill=tk.X, pady=(2, 6))
btn_convex = tk.Button(frame_curv, text="Convex (v)", font=("Helvetica", 8), relief=tk.GROOVE)
btn_convex.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
btn_concave = tk.Button(frame_curv, text="Concave (n)", font=("Helvetica", 8), relief=tk.GROOVE)
btn_concave.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))

lbl_geom = tk.Label(tab_optics, text="GEOMETRY", font=("Helvetica", 8, "bold"))
lbl_geom.pack(anchor="w")
all_label_widgets.append(lbl_geom)

frame_geom = tk.Frame(tab_optics)
frame_geom.pack(fill=tk.X, pady=(2, 6))
btn_circular = tk.Button(frame_geom, text="Circle (r)", font=("Helvetica", 8), relief=tk.GROOVE)
btn_circular.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 1))
btn_parabolic = tk.Button(frame_geom, text="Parab (p)", font=("Helvetica", 8), relief=tk.GROOVE)
btn_parabolic.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
btn_flat = tk.Button(frame_geom, text="Flat (f)", font=("Helvetica", 8), relief=tk.GROOVE)
btn_flat.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(1, 0))

lbl_params = tk.Label(tab_optics, text="PARAMETERS", font=("Helvetica", 8, "bold"))
lbl_params.pack(anchor="w", pady=(2, 0))
all_label_widgets.append(lbl_params)

frame_params = tk.Frame(tab_optics)
frame_params.pack(fill=tk.X, pady=(2, 4))

row_param, lbl_param, ent_param = make_entry_row(frame_params, "Radius 1: ", hyperparameter_val)
row_param.pack(fill=tk.X, pady=1)

row_param2, lbl_param2, ent_param2 = make_entry_row(frame_params, "Radius 2: ", hyperparameter_val2)
row_param2.pack(fill=tk.X, pady=1)

row_thick, lbl_thick, ent_thick = make_entry_row(frame_params, "Thickness: ", thickness_val)
row_thick.pack(fill=tk.X, pady=1)

row_aperture, lbl_aperture, ent_aperture = make_entry_row(frame_params, "Arc / Span: ", mirror_arc_val)
row_aperture.pack(fill=tk.X, pady=1)

row_angle, lbl_angle, ent_angle = make_entry_row(frame_params, "Angle (°): ", angle_val)
row_angle.pack(fill=tk.X, pady=1)

row_air, lbl_air, ent_air = make_entry_row(frame_params, "Air (n): ", air_refractive_index)
row_air.pack(fill=tk.X, pady=1)

row_glass, lbl_glass, ent_glass = make_entry_row(frame_params, "Glass (n): ", glass_refractive_index)
row_glass.pack(fill=tk.X, pady=1)

lbl_angle_hint = tk.Label(tab_optics, text="Angle: (+) anti-clockwise, (-) clockwise", font=("Helvetica", 7, "italic"))
lbl_angle_hint.pack(anchor="w", pady=(1, 4))
all_label_widgets.append(lbl_angle_hint)

# Surface Placement Coordinates & Place Button
frame_place = tk.LabelFrame(tab_optics, text="Target Placement (X, Y)", font=("Helvetica", 8, "bold"), padx=6, pady=4)
frame_place.pack(fill=tk.X, pady=(2, 6))

row_px, lbl_px, ent_place_x = make_entry_row(frame_place, "Target X: ", "10.00")
row_px.pack(fill=tk.X, pady=1)
row_py, lbl_py, ent_place_y = make_entry_row(frame_place, "Target Y: ", "10.00")
row_py.pack(fill=tk.X, pady=1)

btn_place_obj = tk.Button(frame_place, text="Place Object (or Click Canvas)", font=("Helvetica", 8, "bold"), relief=tk.GROOVE)
btn_place_obj.pack(fill=tk.X, pady=(3, 1))

btn_undo = tk.Button(tab_optics, text="Undo (c)", font=("Helvetica", 8, "bold"), relief=tk.GROOVE)
btn_undo.pack(fill=tk.X, pady=(2, 2))

btn_clear = tk.Button(tab_optics, text="Clear All (x)", font=("Helvetica", 8, "bold"), relief=tk.GROOVE)
btn_clear.pack(fill=tk.X, pady=(0, 4))

status_box = tk.Label(
    tab_optics, text="", font=("Helvetica", 8, "italic"),
    justify=tk.LEFT, relief=tk.SOLID, bd=1, padx=6, pady=4, wraplength=270
)
status_box.pack(fill=tk.BOTH, expand=True)

# ------------------------------------------------------------------------------
# TAB 2: Source Settings
# ------------------------------------------------------------------------------
lbl_src_mode = tk.Label(tab_source, text="SOURCE EMISSION TYPE", font=("Helvetica", 8, "bold"))
lbl_src_mode.pack(anchor="w", pady=(0, 4))
all_label_widgets.append(lbl_src_mode)

var_source_type = tk.StringVar(value=source_type)

def on_source_type_change():
    global source_type
    source_type = var_source_type.get()
    update_source_tab_visibility()
    set_modified(True)
    redraw()

rb_point = tk.Radiobutton(tab_source, text="Radial (360° Point Source)", variable=var_source_type, value="point",
                          command=on_source_type_change, font=("Helvetica", 8))
rb_point.pack(anchor="w", pady=1)

rb_parallel = tk.Radiobutton(tab_source, text="Parallel Collimated Beam", variable=var_source_type, value="parallel",
                             command=on_source_type_change, font=("Helvetica", 8))
rb_parallel.pack(anchor="w", pady=1)

frame_src_params = tk.LabelFrame(tab_source, text="Source Beam Parameters", font=("Helvetica", 8, "bold"), padx=6, pady=6)
frame_src_params.pack(fill=tk.X, pady=(8, 8))

row_src_rays, lbl_src_rays, ent_src_rays = make_entry_row(frame_src_params, "Rays Count: ", source_rays_val)
row_src_rays.pack(fill=tk.X, pady=2)

row_src_span, lbl_src_span, ent_src_span = make_entry_row(frame_src_params, "Beam Width: ", source_span_val)
row_src_span.pack(fill=tk.X, pady=2)

row_src_angle, lbl_src_angle, ent_src_angle = make_entry_row(frame_src_params, "Direction (°): ", source_angle_val)
row_src_angle.pack(fill=tk.X, pady=2)

lbl_src_angle_hint = tk.Label(frame_src_params, text="Beam Angle: 0° = +X (Right), 90° = +Y (Up)", font=("Helvetica", 7, "italic"))
lbl_src_angle_hint.pack(anchor="w", pady=(2, 0))
all_label_widgets.append(lbl_src_angle_hint)

frame_src_pos = tk.LabelFrame(tab_source, text="Source Position Coordinates", font=("Helvetica", 8, "bold"), padx=6, pady=6)
frame_src_pos.pack(fill=tk.X, pady=(0, 8))

row_src_x, lbl_src_x, ent_src_x = make_entry_row(frame_src_pos, "X Coord: ", f"{source_pos[0]:.2f}")
row_src_x.pack(fill=tk.X, pady=1)
row_src_y, lbl_src_y, ent_src_y = make_entry_row(frame_src_pos, "Y Coord: ", f"{source_pos[1]:.2f}")
row_src_y.pack(fill=tk.X, pady=1)

def on_update_source():
    global source_rays_val, source_span_val, source_angle_val, source_pos
    try:
        r = int(ent_src_rays.get())
        if r > 0: source_rays_val = r
    except ValueError: pass

    try:
        sp = float(ent_src_span.get())
        if sp > 0: source_span_val = sp
    except ValueError: pass

    try:
        source_angle_val = float(ent_src_angle.get())
    except ValueError: pass

    try:
        x = float(ent_src_x.get())
        y = float(ent_src_y.get())
        source_pos = np.array([x, y])
    except ValueError: pass

    set_modified(True)
    redraw()

btn_apply_src = tk.Button(tab_source, text="Apply Source Settings", font=("Helvetica", 8, "bold"), command=on_update_source, relief=tk.GROOVE)
btn_apply_src.pack(fill=tk.X, pady=(2, 4))

for e in [ent_src_rays, ent_src_span, ent_src_angle, ent_src_x, ent_src_y]:
    e.bind("<Return>", lambda ev: on_update_source())
    e.bind("<FocusOut>", lambda ev: on_update_source())

def update_source_tab_visibility():
    if source_type == 'parallel':
        row_src_span.pack(fill=tk.X, pady=2, after=row_src_rays)
        row_src_angle.pack(fill=tk.X, pady=2, after=row_src_span)
        lbl_src_angle_hint.pack(anchor="w", pady=(2, 0))
    else:
        row_src_span.pack_forget()
        row_src_angle.pack_forget()
        lbl_src_angle_hint.pack_forget()

update_source_tab_visibility()

# ------------------------------------------------------------------------------
# TAB 3: Styling & Colors
# ------------------------------------------------------------------------------
lbl_pal = tk.Label(tab_colors, text="OPTICAL COLOR PALETTE", font=("Helvetica", 8, "bold"))
lbl_pal.pack(anchor="w", pady=(0, 6))
all_label_widgets.append(lbl_pal)

frame_swatches = tk.Frame(tab_colors)
frame_swatches.pack(fill=tk.X)

color_buttons = {}

def pick_color(key):
    global color_source, color_incident, color_reflected, color_refracted, color_mirror, color_lens
    curr = globals()[f"color_{key}"]
    picked = colorchooser.askcolor(color=curr, title=f"Choose Color for {key.capitalize()}")
    if picked and picked[1]:
        hex_c = picked[1]
        globals()[f"color_{key}"] = hex_c
        color_buttons[key].config(bg=hex_c)
        if key == "mirror":
            for s in Surfaces:
                if not s.is_lens: s.color = hex_c
        elif key == "lens":
            for s in Surfaces:
                if s.is_lens: s.color = hex_c
        set_modified(True)
        redraw()

color_items = [
    ("source", "Light Source Marker", color_source),
    ("incident", "Incident Light Rays", color_incident),
    ("reflected", "Reflected Rays (Mirrors/TIR)", color_reflected),
    ("refracted", "Refracted Rays (Lenses)", color_refracted),
    ("mirror", "Mirrors Line Color", color_mirror),
    ("lens", "Lenses Line Color", color_lens)
]

for key, label_text, default_col in color_items:
    row = tk.Frame(frame_swatches)
    row.pack(fill=tk.X, pady=3)
    l_sw = tk.Label(row, text=label_text, font=("Helvetica", 8), anchor="w")
    l_sw.pack(side=tk.LEFT, fill=tk.X, expand=True)
    all_label_widgets.append(l_sw)

    btn_col = tk.Button(row, bg=default_col, width=4, relief=tk.SOLID, bd=1, command=lambda k=key: pick_color(k))
    btn_col.pack(side=tk.RIGHT)
    color_buttons[key] = btn_col

def reset_default_colors():
    global color_source, color_incident, color_reflected, color_refracted, color_mirror, color_lens
    color_source = "#e74c3c"
    color_incident = "#f39c12"
    color_reflected = "#3498db"
    color_refracted = "#2ecc71"
    color_mirror = "#555555"
    color_lens = "#1abc9c"
    for k in color_buttons:
        color_buttons[k].config(bg=globals()[f"color_{k}"])
    for s in Surfaces:
        s.color = color_lens if s.is_lens else color_mirror
    set_modified(True)
    redraw()

btn_reset_colors = tk.Button(tab_colors, text="Reset to Default Colors", font=("Helvetica", 8), command=reset_default_colors, relief=tk.GROOVE)
btn_reset_colors.pack(fill=tk.X, pady=(12, 0))

# ------------------------------------------------------------------------------
# TAB 4: Workbench & File Operations
# ------------------------------------------------------------------------------
lbl_wb_title = tk.Label(tab_file, text="WORKBENCH MANAGEMENT", font=("Helvetica", 8, "bold"))
lbl_wb_title.pack(anchor="w", pady=(0, 6))
all_label_widgets.append(lbl_wb_title)

frame_file_btns = tk.Frame(tab_file)
frame_file_btns.pack(fill=tk.X, pady=(0, 8))

btn_f_new = tk.Button(frame_file_btns, text="New Workbench (Ctrl+N)", font=("Helvetica", 8), relief=tk.GROOVE)
btn_f_new.pack(fill=tk.X, pady=2)

btn_f_open = tk.Button(frame_file_btns, text="Open Workbench... (Ctrl+O)", font=("Helvetica", 8), relief=tk.GROOVE)
btn_f_open.pack(fill=tk.X, pady=2)

btn_f_save = tk.Button(frame_file_btns, text="Save Workbench (Ctrl+S)", font=("Helvetica", 8), relief=tk.GROOVE)
btn_f_save.pack(fill=tk.X, pady=2)

btn_f_saveas = tk.Button(frame_file_btns, text="Save As... (Ctrl+Shift+S)", font=("Helvetica", 8), relief=tk.GROOVE)
btn_f_saveas.pack(fill=tk.X, pady=2)

btn_f_export = tk.Button(frame_file_btns, text="Export Plot Image... (Ctrl+E)", font=("Helvetica", 8, "bold"), relief=tk.GROOVE)
btn_f_export.pack(fill=tk.X, pady=(8, 2))

frame_meta = tk.LabelFrame(tab_file, text="Document Information", font=("Helvetica", 8, "bold"), padx=6, pady=6)
frame_meta.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

lbl_meta_file = tk.Label(frame_meta, text="File: Untitled", font=("Helvetica", 8), anchor="w", wraplength=270, justify=tk.LEFT)
lbl_meta_file.pack(fill=tk.X, pady=1)
all_label_widgets.append(lbl_meta_file)

lbl_meta_status = tk.Label(frame_meta, text="Status: Clean", font=("Helvetica", 8), anchor="w")
lbl_meta_status.pack(fill=tk.X, pady=1)
all_label_widgets.append(lbl_meta_status)

lbl_meta_counts = tk.Label(frame_meta, text="Surfaces: 0", font=("Helvetica", 8), anchor="w")
lbl_meta_counts.pack(fill=tk.X, pady=1)
all_label_widgets.append(lbl_meta_counts)

def update_meta_panel():
    lbl_meta_file.config(text=f"File: {os.path.basename(current_file_path) if current_file_path else 'Untitled'}")
    lbl_meta_status.config(text=f"Status: {'Modified (*)' if is_modified else 'Saved'}")
    lbl_meta_counts.config(text=f"Surfaces: {len(Surfaces)} | Rays: {source_rays_val} ({source_type})")

# ==============================================================================
# File Persistence Logic
# ==============================================================================
def serialize_workbench() -> dict:
    return {
        "version": 1.0,
        "source": {
            "pos": source_pos.tolist(),
            "type": source_type,
            "rays": source_rays_val,
            "span": source_span_val,
            "angle": source_angle_val
        },
        "view": {
            "xlim": view_xlim,
            "ylim": view_ylim
        },
        "environment": {
            "air_refractive_index": air_refractive_index,
            "glass_refractive_index": glass_refractive_index
        },
        "colors": {
            "source": color_source,
            "incident": color_incident,
            "reflected": color_reflected,
            "refracted": color_refracted,
            "mirror": color_mirror,
            "lens": color_lens
        },
        "surfaces": [s.to_dict() for s in Surfaces],
        "history_groups": [
            [Surfaces.index(s) for s in group if s in Surfaces]
            for group in history
        ]
    }

def deserialize_workbench(data: dict):
    global source_pos, source_type, source_rays_val, source_span_val, source_angle_val
    global air_refractive_index, glass_refractive_index
    global color_source, color_incident, color_reflected, color_refracted, color_mirror, color_lens
    global Surfaces, history, view_xlim, view_ylim

    src_data = data.get("source", {})
    source_pos = np.array(src_data.get("pos", [4.0, 10.0]), dtype=float)
    source_type = src_data.get("type", "point")
    var_source_type.set(source_type)
    source_rays_val = src_data.get("rays", 32)
    source_span_val = src_data.get("span", 4.0)
    source_angle_val = src_data.get("angle", 0.0)

    view_data = data.get("view", {})
    view_xlim = view_data.get("xlim", [0.0, 20.0])
    view_ylim = view_data.get("ylim", [0.0, 20.0])

    ent_src_rays.delete(0, tk.END); ent_src_rays.insert(0, str(source_rays_val))
    ent_src_span.delete(0, tk.END); ent_src_span.insert(0, str(source_span_val))
    ent_src_angle.delete(0, tk.END); ent_src_angle.insert(0, str(source_angle_val))
    ent_src_x.delete(0, tk.END); ent_src_x.insert(0, f"{source_pos[0]:.2f}")
    ent_src_y.delete(0, tk.END); ent_src_y.insert(0, f"{source_pos[1]:.2f}")
    update_source_tab_visibility()

    env_data = data.get("environment", {})
    air_refractive_index = env_data.get("air_refractive_index", 1.0)
    glass_refractive_index = env_data.get("glass_refractive_index", 1.5)
    ent_air.delete(0, tk.END); ent_air.insert(0, str(air_refractive_index))
    ent_glass.delete(0, tk.END); ent_glass.insert(0, str(glass_refractive_index))

    cols = data.get("colors", {})
    color_source = cols.get("source", color_source)
    color_incident = cols.get("incident", color_incident)
    color_reflected = cols.get("reflected", color_reflected)
    color_refracted = cols.get("refracted", color_refracted)
    color_mirror = cols.get("mirror", color_mirror)
    color_lens = cols.get("lens", color_lens)

    for k in color_buttons:
        color_buttons[k].config(bg=globals()[f"color_{k}"])

    Surfaces.clear()
    history.clear()
    for s_dict in data.get("surfaces", []):
        Surfaces.append(surface_from_dict(s_dict))

    history_groups = data.get("history_groups", [])
    if history_groups:
        for idx_group in history_groups:
            group = [Surfaces[i] for i in idx_group if i < len(Surfaces)]
            if group: history.append(group)
    else:
        for s in Surfaces:
            history.append([s])

    set_modified(False)
    update_meta_panel()
    redraw()

def load_file_from_path(filepath: str):
    global current_file_path
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        current_file_path = os.path.abspath(filepath)
        deserialize_workbench(data)
    except Exception as e:
        messagebox.showerror("Open Error", f"Could not load file:\n{filepath}\n\nError: {e}")

def on_save():
    global current_file_path
    if not current_file_path:
        on_save_as()
    else:
        try:
            data = serialize_workbench()
            with open(current_file_path, "w") as f:
                json.dump(data, f, indent=2)
            set_modified(False)
            update_meta_panel()
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save file: {e}")

def on_save_as():
    global current_file_path
    path = filedialog.asksaveasfilename(
        defaultextension=".owb",
        filetypes=[("Optical Workbench Files", "*.owb"), ("JSON Files", "*.json"), ("All Files", "*.*")]
    )
    if path:
        current_file_path = path
        on_save()

def on_open():
    global current_file_path
    if is_modified:
        res = messagebox.askyesnocancel("Unsaved Changes", "Save changes to current workbench before opening another?")
        if res is True:
            on_save()
        elif res is None:
            return

    path = filedialog.askopenfilename(
        filetypes=[("Optical Workbench Files", "*.owb;*.json"), ("All Files", "*.*")]
    )
    if path:
        load_file_from_path(path)

def on_new():
    global current_file_path, Surfaces, history
    if is_modified:
        res = messagebox.askyesnocancel("Unsaved Changes", "Save current changes before creating a new workbench?")
        if res is True:
            on_save()
        elif res is None:
            return

    Surfaces.clear()
    history.clear()
    current_file_path = None
    reset_view()
    set_modified(False)
    update_meta_panel()
    redraw()

def on_export_plot():
    path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("PDF Vector", "*.pdf"), ("SVG Vector", "*.svg")]
    )
    if path:
        try:
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
            messagebox.showinfo("Export Successful", f"Plot successfully exported to:\n{path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Could not export plot: {e}")

btn_f_new.config(command=on_new)
btn_f_open.config(command=on_open)
btn_f_save.config(command=on_save)
btn_f_saveas.config(command=on_save_as)
btn_f_export.config(command=on_export_plot)

# Menu Bar
menu_bar = tk.Menu(root)

menu_file = tk.Menu(menu_bar, tearoff=0)
menu_file.add_command(label="New Workbench", accelerator="Ctrl+N", command=on_new)
menu_file.add_command(label="Open Workbench...", accelerator="Ctrl+O", command=on_open)
menu_file.add_command(label="Save Workbench", accelerator="Ctrl+S", command=on_save)
menu_file.add_command(label="Save As...", accelerator="Ctrl+Shift+S", command=on_save_as)
menu_file.add_separator()
menu_file.add_command(label="Export Plot Image...", accelerator="Ctrl+E", command=on_export_plot)
menu_file.add_separator()
menu_file.add_command(label="Exit", command=lambda: on_close())
menu_bar.add_cascade(label="File", menu=menu_file)

menu_edit = tk.Menu(menu_bar, tearoff=0)
menu_edit.add_command(label="Undo Last Surface", accelerator="c / Ctrl+Z", command=lambda: on_undo())
menu_edit.add_command(label="Clear All Surfaces", accelerator="x", command=lambda: on_clear())
menu_bar.add_cascade(label="Edit", menu=menu_edit)

menu_view = tk.Menu(menu_bar, tearoff=0)
menu_view.add_command(label="Toggle Dark Mode", accelerator="Ctrl+D", command=lambda: toggle_theme())
menu_view.add_command(label="Reset View (20x20)", accelerator="Home", command=reset_view)
menu_bar.add_cascade(label="View", menu=menu_view)

root.config(menu=menu_bar)

# ==============================================================================
# Theme Control (True Dark / Light Mode Implementation)
# ==============================================================================
def apply_theme():
    global ACTIVE_BG, INACTIVE_BG
    
    if is_dark_mode:
        bg_main = "#121212"
        bg_panel = "#1e1e1e"
        bg_entry = "#262626"
        border_entry = "#444444"
        fg_main = "#ffffff"
        ACTIVE_BG = '#1e5f48'
        INACTIVE_BG = '#2d2d30'
        btn_action_bg = '#3d2626'
        btn_action_hover = '#4e342e'
        box_bg = '#161616'
        box_border = '#333333'
        btn_theme.config(text="☀️ Light Mode (Ctrl+D)")
        btn_place_obj.config(bg="#1a4231", fg="#ffffff")
        btn_apply_src.config(bg="#1a4231", fg="#ffffff")
        btn_reset_view.config(bg=INACTIVE_BG, fg=fg_main)
    else:
        bg_main = "#eef2f5"
        bg_panel = "#fbfbfb"
        bg_entry = "#ffffff"
        border_entry = "#cccccc"
        fg_main = "#222222"
        ACTIVE_BG = '#a8e6cf'
        INACTIVE_BG = '#f0f0f0'
        btn_action_bg = '#fff0f0'
        btn_action_hover = '#ffd8d8'
        box_bg = '#f7f9fa'
        box_border = '#dddddd'
        btn_theme.config(text="🌙 Dark Mode (Ctrl+D)")
        btn_place_obj.config(bg="#e8f4fd", fg="#000000")
        btn_apply_src.config(bg=ACTIVE_BG, fg="#000000")
        btn_reset_view.config(bg=INACTIVE_BG, fg=fg_main)

    root.configure(bg=bg_main)
    sidebar_container.configure(bg=bg_panel)
    header_frame.configure(bg=bg_panel)
    lbl_app_title.configure(bg=bg_panel, fg=fg_main)
    btn_theme.configure(bg=INACTIVE_BG, fg=fg_main)
    plot_frame.configure(bg=bg_main)
    canvas_widget.configure(bg=bg_main)

    style = ttk.Style()
    style.theme_use('clam')
    style.configure("TNotebook", background=bg_panel, borderwidth=0)
    style.configure("TNotebook.Tab", background=INACTIVE_BG, foreground=fg_main, padding=[10, 4], font=("Helvetica", 8, "bold"), borderwidth=0)
    style.map("TNotebook.Tab", background=[("selected", ACTIVE_BG)], foreground=[("selected", "#ffffff" if is_dark_mode else "#000000")])

    for tab in [tab_optics, tab_source, tab_colors, tab_file]:
        tab.configure(bg=bg_panel)

    for w in [frame_obj, sub_obj, frame_curv, frame_geom, frame_params, frame_file_btns, frame_swatches]:
        w.configure(bg=bg_panel)

    for lf in [frame_place, frame_src_params, frame_src_pos, frame_meta]:
        lf.configure(bg=bg_panel, fg=fg_main, highlightbackground=box_border, highlightcolor=box_border)

    for lbl in all_label_widgets:
        lbl.configure(bg=bg_panel, fg=fg_main)

    for ent in all_entry_widgets:
        ent.configure(bg=bg_entry, fg=fg_main, insertbackground=fg_main, highlightbackground=border_entry, highlightcolor="#3498db")

    rb_point.configure(bg=bg_panel, fg=fg_main, selectcolor=bg_entry, activebackground=bg_panel, activeforeground=fg_main)
    rb_parallel.configure(bg=bg_panel, fg=fg_main, selectcolor=bg_entry, activebackground=bg_panel, activeforeground=fg_main)

    btn_undo.configure(bg=btn_action_bg, activebackground=btn_action_hover, fg=fg_main)
    btn_clear.configure(bg=btn_action_bg, activebackground=btn_action_hover, fg=fg_main)
    btn_reset_colors.configure(bg=INACTIVE_BG, fg=fg_main)
    btn_f_new.configure(bg=INACTIVE_BG, fg=fg_main)
    btn_f_open.configure(bg=INACTIVE_BG, fg=fg_main)
    btn_f_save.configure(bg=INACTIVE_BG, fg=fg_main)
    btn_f_saveas.configure(bg=INACTIVE_BG, fg=fg_main)
    btn_f_export.configure(bg="#2d3748" if is_dark_mode else "#e8f4fd", fg=fg_main)

    status_box.configure(bg=box_bg, fg=fg_main, bd=1, relief=tk.SOLID, highlightbackground=box_border)

    update_ui_state()
    redraw()

def toggle_theme():
    global is_dark_mode
    is_dark_mode = not is_dark_mode
    apply_theme()

btn_theme.config(command=toggle_theme)

# ==============================================================================
# UI State Updates & Callbacks
# ==============================================================================
def update_ui_state():
    fg_col = "#ffffff" if is_dark_mode else "#000000"

    btn_source.config(bg=ACTIVE_BG if selected_category == 'source' else INACTIVE_BG, fg=fg_col)
    btn_mirror.config(bg=ACTIVE_BG if selected_category == 'mirror' else INACTIVE_BG, fg=fg_col)
    btn_lens.config(bg=ACTIVE_BG if selected_category == 'lens' else INACTIVE_BG, fg=fg_col)

    is_flat_item = (selected_shape == 'flat')
    btn_convex.config(bg=ACTIVE_BG if (selected_curvature == 'convex' and not is_flat_item) else INACTIVE_BG, fg=fg_col)
    btn_concave.config(bg=ACTIVE_BG if (selected_curvature == 'concave' and not is_flat_item) else INACTIVE_BG, fg=fg_col)

    btn_circular.config(bg=ACTIVE_BG if selected_shape == 'circular' else INACTIVE_BG, fg=fg_col)
    btn_parabolic.config(bg=ACTIVE_BG if selected_shape == 'parabolic' else INACTIVE_BG, fg=fg_col)
    btn_flat.config(bg=ACTIVE_BG if selected_shape == 'flat' else INACTIVE_BG, fg=fg_col)

    is_lens_mode = (selected_category == 'lens')
    is_source_mode = (selected_category == 'source')
    is_curved_mirror = (selected_category == 'mirror' and selected_shape in ['circular', 'parabolic'])
    is_slab = (is_lens_mode and selected_shape == 'flat')

    if is_lens_mode:
        btn_flat.config(text="Slab (f)")
    else:
        btn_flat.config(text="Flat (f)")

    if is_lens_mode:
        if is_slab:
            row_param2.pack_forget()
            row_thick.pack(fill=tk.X, pady=1, after=row_param)
        else:
            row_param2.pack(fill=tk.X, pady=1, after=row_param)
            row_thick.pack(fill=tk.X, pady=1, after=row_param2)
    else:
        row_param2.pack_forget()
        row_thick.pack_forget()

    if is_curved_mirror:
        row_aperture.pack(fill=tk.X, pady=1, after=row_param)
    else:
        row_aperture.pack_forget()

    if not is_source_mode:
        anchor_row = row_aperture if is_curved_mirror else (row_thick if is_lens_mode else row_param)
        row_angle.pack(fill=tk.X, pady=1, after=anchor_row)
        lbl_angle_hint.pack(anchor="w", pady=(1, 4))
    else:
        row_angle.pack_forget()
        lbl_angle_hint.pack_forget()

    if is_lens_mode:
        if is_slab:
            lbl_param.config(text="Face Width: ")
            lbl_thick.config(text="Thickness: ")
        elif selected_shape == 'circular':
            lbl_param.config(text="Radius 1: ")
            lbl_param2.config(text="Radius 2: ")
            lbl_thick.config(text="Thickness: ")
        else:
            lbl_param.config(text="Focus 1: ")
            lbl_param2.config(text="Focus 2: ")
            lbl_thick.config(text="Thickness: ")
    else:
        if selected_shape == 'circular':
            lbl_param.config(text="Radius: ")
            lbl_aperture.config(text="Arc (°): ")
        elif selected_shape == 'parabolic':
            lbl_param.config(text="Focus: ")
            lbl_aperture.config(text="Span: ")
        else:
            lbl_param.config(text="Length: ")

    angle_note = "Angle: (+) anti-clockwise, (-) clockwise."
    zoom_note = "Zoom: Scroll Wheel (20x20 to 100x100)\nPan: Right-click / Middle-click Drag."
    if selected_category == 'source':
        status_box.config(
            text="Action:\nClick canvas or enter (X, Y) to move source.\n\n"
                 "Hotkeys:\ns, m, l, v, n, r, p, f, c, x\n\n"
                 f"{zoom_note}"
        )
    elif selected_category == 'lens':
        if is_slab:
            status_box.config(
                text=f"Action:\nClick canvas or 'Place Object' for Glass Slab.\n"
                     f"(Face Width = {hyperparameter_val}, Thick = {thickness_val}, Angle = {angle_val}°)\n"
                     f"{angle_note}\n\n{zoom_note}"
            )
        else:
            label = "Radius" if selected_shape == 'circular' else "Focus"
            curv_label = selected_curvature.capitalize()
            status_box.config(
                text=f"Action:\nClick canvas or 'Place Object' for {curv_label} {selected_shape.capitalize()} Lens.\n"
                     f"({label} 1 = {hyperparameter_val}, 2 = {hyperparameter_val2},\n"
                     f"Thick = {thickness_val}, Angle = {angle_val}°)\n"
                     f"{angle_note}\n\n{zoom_note}"
            )
    else:
        if selected_shape == 'flat':
            status_box.config(
                text=f"Action:\nClick canvas or 'Place Object' for Flat Mirror.\n"
                     f"(Length = {hyperparameter_val}, Angle = {angle_val}°)\n"
                     f"{angle_note}\n\n{zoom_note}"
            )
        elif selected_shape == 'circular':
            arc_str = f"±{mirror_arc_val}°" if mirror_arc_val > 0 else "±45°"
            status_box.config(
                text=f"Action:\nClick canvas or 'Place Object' for Circular Mirror.\n"
                     f"(Radius = {hyperparameter_val}, Arc = {arc_str}, Angle = {angle_val}°)\n"
                     f"{angle_note}\n\n{zoom_note}"
            )
        else:
            span_str = f"±{mirror_arc_val}" if mirror_arc_val > 0 else "±1.5"
            status_box.config(
                text=f"Action:\nClick canvas or 'Place Object' for Parabolic Mirror.\n"
                     f"(Focus = {hyperparameter_val}, Span = {span_str}, Angle = {angle_val}°)\n"
                     f"{angle_note}\n\n{zoom_note}"
            )
    update_meta_panel()

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

btn_source.config(command=lambda: set_category('source'))
btn_mirror.config(command=lambda: set_category('mirror'))
btn_lens.config(command=lambda: set_category('lens'))

btn_convex.config(command=lambda: set_curvature('convex'))
btn_concave.config(command=lambda: set_curvature('concave'))

btn_circular.config(command=lambda: set_shape('circular'))
btn_parabolic.config(command=lambda: set_shape('parabolic'))
btn_flat.config(command=lambda: set_shape('flat'))

def sync_params(event=None):
    global hyperparameter_val, hyperparameter_val2, thickness_val, mirror_arc_val, angle_val, air_refractive_index, glass_refractive_index
    try:
        val = float(ent_param.get())
        if val > 0: hyperparameter_val = val
    except ValueError: pass

    try:
        val = float(ent_param2.get())
        if val > 0: hyperparameter_val2 = val
    except ValueError: pass

    try:
        val = float(ent_thick.get())
        if val > 0: thickness_val = val
    except ValueError: pass

    try:
        val = float(ent_aperture.get())
        if val >= 0:
            mirror_arc_val = val
    except ValueError: pass

    try:
        angle_val = float(ent_angle.get())
    except ValueError: pass

    try:
        val = float(ent_air.get())
        if val > 0: air_refractive_index = val
    except ValueError: pass

    try:
        val = float(ent_glass.get())
        if val > 0:
            glass_refractive_index = val
            for s in Surfaces:
                if s.is_lens:
                    s.refractive_index = glass_refractive_index
    except ValueError: pass

    update_ui_state()
    redraw()

for entry in [ent_param, ent_param2, ent_thick, ent_aperture, ent_angle, ent_air, ent_glass]:
    entry.bind("<Return>", sync_params)
    entry.bind("<FocusOut>", sync_params)

def on_undo():
    if history:
        last_group = history.pop()
        for item in last_group:
            if item in Surfaces:
                Surfaces.remove(item)
        set_modified(True)
        redraw()

def on_clear():
    Surfaces.clear()
    history.clear()
    set_modified(True)
    redraw()

btn_undo.config(command=on_undo)
btn_clear.config(command=on_clear)

# Keyboard Shortcuts Dispatcher
def on_key_press(event):
    focused = root.focus_get()
    if isinstance(focused, tk.Entry):
        return

    ctrl_pressed = (event.state & 0x4) != 0

    if ctrl_pressed:
        k = event.keysym.lower()
        if k == 's':
            if (event.state & 0x1) != 0:
                on_save_as()
            else:
                on_save()
        elif k == 'o': on_open()
        elif k == 'n': on_new()
        elif k == 'e': on_export_plot()
        elif k == 'z': on_undo()
        elif k == 'd': toggle_theme()
        return

    if event.keysym == 'Home':
        reset_view()
        return

    if not event.char:
        return
    k = event.char.lower()
    if k == 's': set_category('source')
    elif k == 'm': set_category('mirror')
    elif k == 'l': set_category('lens')
    elif k == 'v': set_curvature('convex')
    elif k == 'n': set_curvature('concave')
    elif k == 'r': set_shape('circular')
    elif k == 'p': set_shape('parabolic')
    elif k == 'f': set_shape('flat')
    elif k == 'c': on_undo()
    elif k == 'x': on_clear()

root.bind("<Key>", on_key_press)
root.bind("<Control-s>", lambda e: on_save())
root.bind("<Control-S>", lambda e: on_save_as())
root.bind("<Control-o>", lambda e: on_open())
root.bind("<Control-n>", lambda e: on_new())
root.bind("<Control-e>", lambda e: on_export_plot())
root.bind("<Control-z>", lambda e: on_undo())
root.bind("<Control-d>", lambda e: toggle_theme())
root.bind("<Home>", lambda e: reset_view())

# ==============================================================================
# Unified Surface Placement Engine
# ==============================================================================
def place_surface_at(pos: np.ndarray):
    global source_pos
    sync_params()
    angle_rad = math.radians(angle_val)
    axis_dir = np.array([math.cos(angle_rad), math.sin(angle_rad)])

    if selected_category == 'source':
        source_pos = pos
        ent_src_x.delete(0, tk.END); ent_src_x.insert(0, f"{source_pos[0]:.2f}")
        ent_src_y.delete(0, tk.END); ent_src_y.insert(0, f"{source_pos[1]:.2f}")
        set_modified(True)
        redraw()

    elif selected_category == 'mirror':
        is_concave = (selected_curvature == 'concave')

        if selected_shape == 'circular':
            arc_limit = mirror_arc_val if mirror_arc_val > 0 else 45.0
            if is_concave:
                # Concave mirror: cavity faces incoming rays (-axis_dir), converges light
                centre = pos - hyperparameter_val * axis_dir
                mirror = Circle(centre, hyperparameter=hyperparameter_val, is_concave=False, angle=angle_rad, parameter=[-arc_limit, arc_limit], color=color_mirror)
            else:
                # Convex mirror: dome faces incoming rays (-axis_dir), diverges light
                centre = pos + hyperparameter_val * axis_dir
                mirror = Circle(centre, hyperparameter=hyperparameter_val, is_concave=True, angle=angle_rad, parameter=[-arc_limit, arc_limit], color=color_mirror)

        elif selected_shape == 'parabolic':
            span_limit = mirror_arc_val if mirror_arc_val > 0 else 1.5
            mirror = Parabola(pos, hyperparameter=hyperparameter_val, is_concave=is_concave, angle=angle_rad, parameter=[-span_limit, span_limit], color=color_mirror)

        else:  # Flat mirror
            length = hyperparameter_val
            surf_dir = np.array([-math.sin(angle_rad), math.cos(angle_rad)])
            p_start = pos - (length/2) * surf_dir
            mirror = Line(p_start, surf_dir, is_concave=False, angle=angle_rad, parameter=length, is_lens=False, color=color_mirror)

        Surfaces.append(mirror)
        history.append([mirror])
        set_modified(True)
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
            color=color_lens
        )
        if selected_shape == 'flat':
            new_surfs = lens.draw_slab()
        elif selected_curvature == 'convex':
            new_surfs = lens.draw_convex()
        else:
            new_surfs = lens.draw_concave()

        Surfaces.extend(new_surfs)
        history.append(new_surfs)
        set_modified(True)
        redraw()

# ==============================================================================
# Mouse Interaction: Canvas Clicks, Scroll Zoom, Drag Pan
# ==============================================================================
def on_mouse_press(event):
    global pan_active, pan_start_data
    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        return
    
    # Right-click (3) or Middle-click (2) to pan
    if event.button in (2, 3):
        pan_active = True
        pan_start_data = (event.x, event.y, list(ax.get_xlim()), list(ax.get_ylim()))
    elif event.button == 1:
        # Left-click to place objects or move source
        pos = np.array([event.xdata, event.ydata])
        ent_place_x.delete(0, tk.END); ent_place_x.insert(0, f"{pos[0]:.2f}")
        ent_place_y.delete(0, tk.END); ent_place_y.insert(0, f"{pos[1]:.2f}")
        place_surface_at(pos)

def on_mouse_motion(event):
    global view_xlim, view_ylim
    if not pan_active or pan_start_data is None:
        return
    
    x0, y0, orig_xlim, orig_ylim = pan_start_data
    dx = event.x - x0
    dy = event.y - y0
    
    bbox = ax.get_window_extent()
    data_width = orig_xlim[1] - orig_xlim[0]
    data_height = orig_ylim[1] - orig_ylim[0]
    
    shift_x = -(dx / bbox.width) * data_width
    shift_y = -(dy / bbox.height) * data_height

    view_xlim = [orig_xlim[0] + shift_x, orig_xlim[1] + shift_x]
    view_ylim = [orig_ylim[0] + shift_y, orig_ylim[1] + shift_y]
    ax.set_xlim(view_xlim)
    ax.set_ylim(view_ylim)
    canvas.draw_idle()

def on_mouse_release(event):
    global pan_active, pan_start_data
    if event.button in (2, 3):
        pan_active = False
        pan_start_data = None

def on_scroll(event):
    global view_xlim, view_ylim
    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        return

    cur_xlim = list(ax.get_xlim())
    cur_ylim = list(ax.get_ylim())
    cur_span = cur_xlim[1] - cur_xlim[0]

    # Zoom in or out factor
    if event.button == 'up' or (hasattr(event, 'step') and event.step > 0):
        scale = 1.0 / 1.15  # Zoom in
    elif event.button == 'down' or (hasattr(event, 'step') and event.step < 0):
        scale = 1.15        # Zoom out
    else:
        return

    new_span = cur_span * scale

    # Clamping range strictly between 20x20 and 100x100
    if new_span < MIN_SPAN:
        new_span = MIN_SPAN
    elif new_span > MAX_SPAN:
        new_span = MAX_SPAN

    if abs(new_span - cur_span) < 1e-5:
        return

    # Center zoom around current mouse cursor coordinate
    rel_x = (event.xdata - cur_xlim[0]) / cur_span
    rel_y = (event.ydata - cur_ylim[0]) / cur_span

    view_xlim = [event.xdata - rel_x * new_span, event.xdata + (1.0 - rel_x) * new_span]
    view_ylim = [event.ydata - rel_y * new_span, event.ydata + (1.0 - rel_y) * new_span]

    ax.set_xlim(view_xlim)
    ax.set_ylim(view_ylim)
    canvas.draw_idle()

fig.canvas.mpl_connect('button_press_event', on_mouse_press)
fig.canvas.mpl_connect('motion_notify_event', on_mouse_motion)
fig.canvas.mpl_connect('button_release_event', on_mouse_release)
fig.canvas.mpl_connect('scroll_event', on_scroll)

def on_place_button_click():
    try:
        x = float(ent_place_x.get())
        y = float(ent_place_y.get())
        place_surface_at(np.array([x, y]))
    except ValueError:
        messagebox.showerror("Coordinate Error", "Please enter valid numeric X and Y coordinates.")

btn_place_obj.config(command=on_place_button_click)

def on_close():
    if is_modified:
        res = messagebox.askyesnocancel("Unsaved Changes", "Save changes to current optical workbench before closing?")
        if res is True:
            on_save()
        elif res is None:
            return
    root.quit()
    root.destroy()
    plt.close('all')

root.protocol("WM_DELETE_WINDOW", on_close)

# ==============================================================================
# Initialization & Direct File Opening Support (sys.argv)
# ==============================================================================
apply_theme()

if len(sys.argv) > 1:
    init_path = sys.argv[1]
    if os.path.isfile(init_path):
        load_file_from_path(init_path)

root.mainloop()