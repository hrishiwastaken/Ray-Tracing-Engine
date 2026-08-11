import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.widgets import TextBox
import math
from typing import List
# Initializing Matplotlib
fig, ax = plt.subplots(figsize=(8, 8))
box_ax = None
textbox = None

# Function to reset and format the axes upon every click
def setup_axes():
    ax.set_aspect("equal")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title("Recursive Ray Tracing \n Left Click to move Source, Right Click to place mirror, Shift + Right Click to place inverted mirror, C to undo mirror, Control + Right Click to place lens.")

# Mirror Object(also acts as a Lens if modifier flag is turned on)
class Mirror:
    def __init__(self, centre, radius, aperture, facing, lens=False):
        self.centre = np.array(centre, dtype="float")
        self.radius = radius
        self.angle = aperture    
        self.facing = facing   
        self.centre_angle = 0 if self.facing == 1 else 180
        self.lens = lens
    
    def draw(self):
        t1 = self.centre_angle - self.angle/2
        t2 = self.centre_angle + self.angle/2        
        arc = Arc(
            tuple(self.centre), width=2*self.radius, height=2*self.radius,
            angle=0, theta1=t1,theta2=t2,
            linewidth=3, color='black'
        )
        ax.add_patch(arc)

  
    def normal(self, point) -> np.ndarray:
        return np.array(point, dtype="float") - self.centre   # Uses vectoral analysis and properties of circle to compute normal.
# Line Segment Object
class LineSegment:
    def __init__(self, p1, p2, color="black", linewidth=3):
        self.p1 = p1
        self.p2 = p2
        self.color = color
        self.lw = linewidth
        self.lens = False
        # Dummy attributes
        self.centre = np.array([0.0, 0.0])
        self.radius = 0.0
        self.angle = 0
        self.centre_angle = 0
    
    def draw(self):
        ax.plot([self.p1[0], self.p2[0]], [self.p1[1], self.p2[1]], color=self.color, lw=self.lw)

# The Lens Object( Uses Mirrors as a helper class )
class Lens:
    def __init__(self, origin, radius1, radius2, thickness):
        self.origin = np.array(origin, dtype="float")
        self.r1 = radius1
        self.r2 = radius2
        self.thickness = thickness
        # Distance between centres of circles
        self.D = self.r1 + self.r2 - self.thickness
        self.a = ((self.D) ** 2 + (self.r1) ** 2 - (self.r2) ** 2) / (
            2 * self.D
        )
        self.b = self.D - self.a

        self.h = (abs((self.r1) ** 2 - self.a**2)) ** 0.5
        self.phi = math.degrees(2 * math.atan2(self.h, self.a))
        self.theita = math.degrees(2 * math.atan2(self.h, self.b))

    def draw_convex(self):
        lens_obj = []
        i1 = self.a * np.array([1.0, 0.0])
        i2 = self.b * np.array([1.0, 0.0])
        # Surface 1 (Left surface: bulges left towards source)
        lens_obj.append(
            Mirror(
                centre=self.origin + i1,
                radius=self.r1,
                aperture=self.phi,
                facing=-1,
                lens=True,
            )
        )
        # Surface 2 (Right surface: bulges right away from source)
        lens_obj.append(
            Mirror(
                centre=self.origin - i2,
                radius=self.r2,
                aperture=self.theita,
                facing=1,
                lens=True,
            )
        )
        return lens_obj

    def draw_biconcave(self):
            lens_obj = []
            i1 = (self.a+self.thickness) * np.array([1.0, 0.0])
            i2 = (self.b+self.thickness) * np.array([1.0, 0.0])
            c1 = self.origin - i1
            c2 = self.origin + i2
            # Surface 1 (Left surface: bulges left towards source)
            lens_obj.append(
                Mirror(
                    centre=c1,
                    radius=self.r1,
                    aperture=self.phi,
                    facing=1,
                    lens=True,
                )
            )
            # Surface 2 (Right surface: bulges right away from source)
            lens_obj.append(
                Mirror(
                    centre=c2,
                    radius=self.r2,
                    aperture=self.theita,
                    facing=-1,
                    lens=True,
                )
            )
            phi_rad = math.radians(self.phi)
            theita_rad = math.radians(self.theita)

            d1 = np.array([math.cos(phi_rad/2), math.sin(phi_rad/2)])
            d2 = np.array([-math.cos(theita_rad/2), math.sin(theita_rad/2)])
            p1t = c1 + d1*self.r1
            p2t = c2 + d2*self.r2
            lens_obj.append(LineSegment(p1t, p2t))
            d1[1] *= -1
            d2[1] *= -1
            p1b = c1 + d1*self.r1
            p2b = c2 + d2*self.r2
            lens_obj.append(LineSegment(p1b, p2b))
            return lens_obj
            

# The ray object containing all ray properties and methods

class Ray:
    def __init__(self, origin, direction, mirrors, bounces=10, color="orange", in_glass=False):
        self.origin = np.array(origin, dtype=float)
        self.direction = direction / np.linalg.norm(direction)
        self.length = 20
        self.mirrors: List[Mirror] = mirrors
        self.bounces = bounces  #---Ray Bounce limit to prevent infinite bouncing
        self.color = color      #---Ray Colour
        self.in_glass = in_glass
        
        # Default parametric ray equation
        self.end = self.origin + self.length * self.direction
        
        

        # Checking if ray intersects with any mirror
        self.points = []
        for mirror in self.mirrors:
            # Aperture Check for each mirrors
            t_intersections = self.nature(mirror)
            if not t_intersections:
                pass 
            else:
                t_valid = self.parameter(t_intersections, mirror)
                if t_valid is None:
                    pass
                else:
                    self.points.append((t_valid, mirror))
        # Bounce check
        if self.bounces <= 0:
            pass
        else:
            if self.points != []:
                t, mirror = min(self.points, key=lambda x: x[0])  # Finds point of intersection closest to ray, i.e the mirror which the ray intersects first, to prevent ghost rays
                self.end = self.origin + t*self.direction
                self.draw()
                self.draw_reflection_or_refraction(mirror) 
            else:
                self.draw()
            return
                
                
    # Quadratic Solver to find points of intersection
    def nature(self, mirror):
        f = self.origin[0] - mirror.centre[0]
        g = self.origin[1] - mirror.centre[1]
        
        Beta = f * self.direction[0] + g * self.direction[1]
        Delta = f**2 + g**2 - mirror.radius**2
        discriminant = 4 * (Beta**2) - 4 * Delta
        
        if discriminant >= 0:                           # Using Quadratic Discriminant to check for intersection
            t1 = (-2 * Beta + discriminant**0.5) / 2
            t2 = (-2 * Beta - discriminant**0.5) / 2
            valid_t = [t for t in (t1, t2) if t > 1e-5] 
            return sorted(valid_t)
        return [] 
        
    # Aperture Check function by checking parametric co-ordinate of circle range
    def parameter(self, pts, mirror):
            for pt in pts:
                intersection_point = self.origin + pt * self.direction
                r = intersection_point - mirror.centre
                # Fixed lower half of mirrror not working properly for concave curve(Angle Normalization Error)
                angle_deg = math.degrees(math.atan2(r[1], r[0]))
                diff = (angle_deg - mirror.centre_angle + 180) % 360 - 180
                if abs(diff) <= mirror.angle / 2:
                    return pt
            return None
        
    # Drawing source rays
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
        
    # Drawing Reflected Rays with recursive reflections
    def draw_reflection_or_refraction(self, mirror):
        incident_dir = self.direction
        normal = mirror.normal(self.end)
        normal /= np.linalg.norm(normal)

        # Standard Mirror Reflection
        if not mirror.lens:
            dot = np.dot(incident_dir, normal)
            reflect_dir = incident_dir - 2 * dot * normal

            Ray(
                origin=self.end,
                direction=reflect_dir,
                mirrors=self.mirrors,
                bounces=self.bounces - 1,
                color="blue",
            )

        # Lens Refraction (Snell's Law)
        else:
            dot = np.dot(incident_dir, normal)

            # Determine whether entering or exiting glass
            if dot < 0:
                oriented_normal = normal
            else:
                oriented_normal = -normal
            if not self.in_glass:
                n1, n2 = 1.0, 1.5  # Air -> Glass
            else:
                n1, n2 = 1.5, 1.0  # Glass -> Air

            n = n1 / n2
            cos_i = -np.dot(incident_dir, oriented_normal)   # The dot product
            sin2_t = (n**2) * (1.0 - cos_i**2)  # Used for checking total internal reflection(negative root case in the vector form of snell's law)

            # Total Internal Reflection check
            if sin2_t > 1.0:
                refracted_dir = (
                    incident_dir
                    - 2 * np.dot(incident_dir, oriented_normal) * oriented_normal
                )
                glasscheck = self.in_glass
            else:

                cos_t = math.sqrt(1.0 - sin2_t)
                refracted_dir = (
                    n * incident_dir + (n * cos_i - cos_t) * oriented_normal
                )
                glasscheck = not self.in_glass

            Ray(
                origin=self.end,
                direction=refracted_dir,
                mirrors=self.mirrors,
                bounces=self.bounces - 1,
                color="blue",
                in_glass = glasscheck,
            )

        
# Defining the source object, main light emitter.
class Source:
    def __init__(self, origin, mirrors: Mirror, rays):
        self.rays = []
        self.origin = np.array(origin, dtype=float)
        self.mirrors = mirrors
        n = 360/rays
        
        for i in range(rays):
            a = np.radians(i * n)
            d = np.array([np.cos(a), np.sin(a)])
            self.rays.append(Ray(self.origin, d, self.mirrors))
            
        ax.scatter(self.origin[0], self.origin[1], color="red", zorder=5, label="Source")


# Initializing Default Objects
mirrors = [
    Mirror(centre=[10, 10], radius=6, aperture=70, facing=1),
    Mirror(centre=[5, 5], radius=6, aperture=70, facing=1)
]

source_pos = np.array([10.0, 10.0])


# Render function that handles clearing and drawing
def draw_scene():
    ax.cla()
    setup_axes()

    for mirror in mirrors:
        mirror.draw()

    Source(
        origin=source_pos,
        mirrors=mirrors,
        rays=50
    )

    ax.legend(loc="upper right")
    fig.canvas.draw()


# Mouse + Keyboard  Input Handler
def close_input():
    global box_ax, textbox
    if box_ax is not None:
        box_ax.remove()
        box_ax = None
        textbox = None
        fig.canvas.draw_idle()

def create_textbox(event, label, callback):
    global box_ax, textbox
    if box_ax is not None:
        box_ax.remove()

    fig_x, fig_y = fig.transFigure.inverted().transform((event.x, event.y))
    box_width = 0.3
    box_height = 0.05
    fig_x = min(fig_x, 1 - box_width)
    fig_y = min(fig_y, 1 - box_height)

    box_ax = fig.add_axes([fig_x, fig_y, box_width, box_height])
    textbox = TextBox(box_ax, label)
    textbox.on_submit(callback)
    fig.canvas.draw_idle()

# Mouse Input handler
def on_click(event):
    global box_ax, textbox

    if event.inaxes is None or (box_ax is not None and event.inaxes is box_ax):
        return

    # Normalize pressed keys for reliable modifier checking
    key = event.key.lower() if event.key else ""
    has_ctrl = "ctrl" in key or "control" in key
    has_shift = "shift" in key

    # Assigning Left click for moving source
    if event.button == 1:
        if box_ax is not None:
            close_input()
        source_pos[:] = [event.xdata, event.ydata]
        draw_scene()

    # Assigning Right Click for placing mirror or lens
    elif event.button == 3:
        pole = np.array([event.xdata, event.ydata])

        # Holding Control to place lens
        if has_ctrl:
            # Save whether Shift was held at the moment of the click
            is_biconcave = has_shift

            values = []
            prompts = [
                "Radius 1: ",
                "Radius 2: ",
                "Thickness of Lens: ",
            ]

            def submit_lens(text):
                try:
                    values.append(float(text))
                except ValueError:
                    textbox.set_val("")
                    return

                if len(values) < len(prompts):
                    create_textbox(event, prompts[len(values)], submit_lens)
                else:
                    lens = Lens(pole, values[0], values[1], values[2])

                    # Ctrl + Shift + Right Click gives Biconcave Lens
                    if is_biconcave:
                        lens_helper_mirror = lens.draw_biconcave()
                    # Ctrl + Right Click gives Convex Lens
                    else:
                        lens_helper_mirror = lens.draw_convex()

                    mirrors.extend(lens_helper_mirror)
                    close_input()
                    draw_scene()

            create_textbox(event, prompts[0], submit_lens)

        # All mirror placements ask for radius (Shift toggles facing direction)
        else:
            facing = -1 if has_shift else 1

            def submit_mirror(text):
                try:
                    radius = float(text)
                except ValueError:
                    textbox.set_val("")
                    return

                centre = pole - np.array([facing * radius, 0])
                mirrors.append(
                    Mirror(
                        centre=centre,
                        radius=radius,
                        aperture=70,
                        facing=facing,
                    )
                )
                close_input()
                draw_scene()

            create_textbox(event, "Radius: ", submit_mirror)
# Keyboard handler
def on_key(event):
    if event.key.lower() == "c": 
        if mirrors:
            pop = mirrors.pop()
            if pop.lens==True and mirrors[-1].lens == True:         # Remove Consecutive Mirror objects with lens modifier true to remove entire lens with one button press
                mirrors.pop()
            draw_scene()


# Connecting events
fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("key_press_event", on_key)


# Initial draw
draw_scene()

plt.show()