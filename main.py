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
    ax.set_title("Recursive Ray Tracing (Click anywhere to move the Source)")

# The mirror object containing all mirror properties
class Mirror:
    def __init__(self, centre, radius, aperture):
        self.centre = np.array(centre, dtype="float")
        self.radius = radius
        self.angle = aperture       

        arc = Arc(
            tuple(self.centre), width=2*self.radius, height=2*self.radius,
            angle=0, theta1=-self.angle/2,theta2=self.angle/2,
            linewidth=3, color='black'
        )
        ax.add_patch(arc)
    
    def normal(self, point) -> np.ndarray:
        return self.centre - np.array(point, dtype="float") 

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
        
        

        # Finding intersections with sphere of mirror 
        
        for mirror in self.mirrors:
            # Aperture Check for each mirrors
            t_intersections = self.nature(mirror)
            if not t_intersections:
                self.draw() 
            else:
                t_valid = self.parameter(t_intersections, mirror)
                if t_valid is None:
                    self.draw() 
                else:
                    self.end = self.origin + t_valid * self.direction
                    # Bounce check
                    if self.bounces <= 0:
                        pass
                    else:
                        self.draw()
                        self.draw_reflection(mirror) 
                        return
                
                
    # Quadratic Solver to find points of intersection
    def nature(self, mirror):
        f = self.origin[0] - mirror.centre[0]
        g = self.origin[1] - mirror.centre[1]
        
        Beta = f * self.direction[0] + g * self.direction[1]
        Delta = f**2 + g**2 - mirror.radius**2
        discriminant = 4 * (Beta**2) - 4 * Delta
        
        if discriminant >= 0:
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
            if -mirror.angle/2 <= angle_deg <= mirror.angle/2:
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
        
        dot = np.dot(incident_dir, normal)
        reflect_dir = incident_dir - 2 * dot * normal
        
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

# Render function that handles clearing and drawing
def draw_scene(origin_x, origin_y):
    ax.cla()       # Clear the current axes to remove old rays
    setup_axes()   # Re-apply limits, grid, and title
    
    mirrors = [Mirror(centre=[10, 10], radius=6, aperture=70), Mirror(centre=[5, 5], radius=6, aperture=70)]
    source = Source(origin=[origin_x, origin_y], mirrors=mirrors, rays=6)  
    ax.legend(loc="upper right")
    fig.canvas.draw() # Force matplotlib to update the canvas

# Event handler for mouse clicks
def on_click(event):
    # Ensure the click is inside the plot boundaries
    if event.inaxes is not None:
        draw_scene(event.xdata, event.ydata)
    

# Connect the click event to our handler function
fig.canvas.mpl_connect('button_press_event', on_click)

# Initial draw
reflections: list[Ray] = np.array([])
draw_scene(10, 10)

plt.show()