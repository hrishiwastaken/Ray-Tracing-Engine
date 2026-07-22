import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import math
from typing import List
# Initializing Matplotlib
fig, ax = plt.subplots(figsize=(8, 8))

# Function to reset and format the axes upon every click
def setup_axes():
    ax.set_aspect("equal")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title("Recursive Ray Tracing \n Left Click to move Source, Right Click to place mirror, Shift + Right Click to place inverted mirror, C to undo mirror")

# The mirror object containing all mirror properties
class Mirror:
    def __init__(self, centre, radius, aperture, facing):
        self.centre = np.array(centre, dtype="float")
        self.radius = radius
        self.angle = aperture    
        self.facing = facing   
        self.centre_angle = 0 if self.facing == 1 else 180
    
    def draw(self):
        if self.facing == 1:
            t1 = -self.angle/2
            t2 = self.angle/2
        else:
            t1 = 180 - self.angle/2
            t2 = 180 + self.angle/2
        
        arc = Arc(
            tuple(self.centre), width=2*self.radius, height=2*self.radius,
            angle=0, theta1=t1,theta2=t2,
            linewidth=3, color='black'
        )
        ax.add_patch(arc)
    
    def normal(self, point) -> np.ndarray:
        return self.centre - np.array(point, dtype="float")   # Uses vectoral analysis and properties of circle to compute normal.

# The ray object containing all ray properties and methods

class Ray:
    def __init__(self, origin, direction, mirrors, bounces=5, color="orange"):
        self.origin = np.array(origin, dtype=float)
        self.direction = direction / np.linalg.norm(direction)
        self.length = 20
        self.mirrors: List[Mirror] = mirrors
        self.bounces = bounces  #---Ray Bounce limit to prevent infinite bouncing
        self.color = color      #---Ray Colour
        
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
                self.draw_reflection(mirror) 
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
            
            angle_deg = math.degrees(math.atan2(r[1], r[0]))
            if angle_deg < 0:
                angle_deg += 360                         # Normalizing the Angle

            if mirror.centre_angle - mirror.angle/2 <= angle_deg <= mirror.centre_angle + mirror.angle/2:        # Checking if ray passes through aperture or just sphere of mirror
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
    def draw_reflection(self, mirror):
        incident_dir = self.direction 
        normal = mirror.normal(self.end)
        normal /= np.linalg.norm(normal)
        if mirror.facing == -1:
            normal *= -1
        
        dot = np.dot(incident_dir, normal)
        reflect_dir = incident_dir - 2 * dot * normal   # Vector form of reflection, just flipping the component along the normal to maintain same angle but inverted direction.
        
        Ray(
            origin=self.end, 
            direction=reflect_dir, 
            mirrors=self.mirrors, 
            bounces=self.bounces - 1, 
            color="blue"
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
        rays=18
    )

    ax.legend(loc="upper right")
    fig.canvas.draw()


# Mouse Input handler
def on_click(event):
    if event.inaxes is None:
        return

    # Assigning Left click for moving source
    if event.button == 1:
        source_pos[:] = [event.xdata, event.ydata]
        draw_scene()

    # Assigning Right Click for placing mirror
    elif event.button == 3:
        pole = np.array([event.xdata, event.ydata])

        radius = 6

        # Holding shift to place mirror
        facing = -1 if event.key == "shift" else 1

        centre = pole - np.array([facing * radius, 0])

        mirrors.append(
            Mirror(
                centre=centre,
                radius=radius,
                aperture=70,
                facing=facing
            )
        )

        draw_scene()


# Keyboard handler
def on_key(event):
    if event.key.lower() == "c":
        if mirrors:
            mirrors.pop()
            draw_scene()


# Connecting events
fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("key_press_event", on_key)


# Initial draw
reflections: list[Ray] = np.array([])
draw_scene()

plt.show()