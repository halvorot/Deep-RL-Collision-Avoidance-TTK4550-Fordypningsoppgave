import numpy as np
import shapely.geometry
import shapely.affinity
import gym_auv.utils.geomutils as geom
from abc import ABC, abstractmethod
import copy

class BaseObstacle(ABC):
    def __init__(self, *args, **kwargs) -> None:
        """Initializes obstacle instance by calling private setup method implemented by
         subclasses of BaseObstacle and calculating obstacle boundary."""
        self._prev_position = []
        self._prev_heading = []
        self._setup(*args, **kwargs)
        self._boundary = self._calculate_boundary()
        if not self._boundary.is_valid:
            self._boundary = self._boundary.buffer(0)
        self._init_boundary = copy.deepcopy(self._boundary)


    @property
    def boundary(self) -> shapely.geometry.Polygon:
        """shapely.geometry.Polygon object used for simulating the 
        sensors' detection of the obstacle instance."""
        return self._boundary

    @property
    def init_boundary(self) -> shapely.geometry.Polygon:
        """shapely.geometry.Polygon object used for simulating the 
        sensors' detection of the obstacle instance."""
        return self._init_boundary

    def update(self, dt:float) -> None:
        """Updates the obstacle according to its dynamic behavior, e.g. 
        a ship model and recalculates the boundary."""
        has_changed = self._update(dt)
        if has_changed:
            self._boundary = self._calculate_boundary()
            if not self._boundary.is_valid:
                self._boundary = self._boundary.buffer(0)

    @abstractmethod
    def _calculate_boundary(self) -> shapely.geometry.Polygon:
        """Returns a shapely.geometry.Polygon instance representing the obstacle
        given its current state."""

    @abstractmethod
    def _setup(self, *args, **kwargs) -> None:
        """Initializes the obstacle given the constructor parameters provided to
        the specific BaseObstacle extension."""

    def _update(self, _dt:float) -> bool:
        """Performs the specific update routine associated with the obstacle.
        Returns a boolean flag representing whether something changed or not.

        Returns
        -------
        has_changed : bool
        """
        return False

    @property
    def path_taken(self) -> list:
        """Returns an array holding the path of the obstacle in cartesian
        coordinates."""
        return self._prev_position

    @property
    def heading_taken(self) -> list:
        """Returns an array holding the heading of the obstacle at previous timesteps."""
        return self._prev_heading

class CircularObstacle(BaseObstacle):
    def _setup(self, position, radius, color=(0.6, 0, 0)):
        self.color = color
        if not isinstance(position, np.ndarray):
            position = np.array(position)
        if radius < 0:
            raise ValueError
        self.static = True
        self.radius = radius
        self.position = position.flatten()

    def _calculate_boundary(self):
        return shapely.geometry.Point(*self.position).buffer(self.radius).boundary.simplify(0.3, preserve_topology=False)

class PolygonObstacle(BaseObstacle):
    def _setup(self, points, color=(0.6, 0, 0)):
        self.static = True
        self.color = color
        self.points = points

    def _calculate_boundary(self):
        return shapely.geometry.Polygon(self.points)

class LineObstacle(BaseObstacle):
    def _setup(self, points):
        self.static = True
        self.points = points

    def _calculate_boundary(self):
        return shapely.geometry.LineString(self.points)

class VesselObstacle(BaseObstacle):
    def _setup(self, width, trajectory, init_position=None, init_heading=None, init_update=True, name=''):
        self.static = False
        self.width = width
        self.trajectory = trajectory
        self.trajectory_velocities = []
        self.name = name
        i = 0
        while i < len(trajectory)-1:
            cur_t = trajectory[i][0]
            next_t = trajectory[i+1][0]
            cur_waypoint = trajectory[i][1]
            next_waypoint = trajectory[i+1][1]

            dx = (next_waypoint[0] - cur_waypoint[0])/(next_t - cur_t)
            dy = (next_waypoint[1] - cur_waypoint[1])/(next_t - cur_t)

            for _ in range(cur_t, next_t):
                self.trajectory_velocities.append((dx, dy))
            
            i+= 1

        self.waypoint_counter = 0
        self.points = [
            (-self.width/2, -self.width/2),
            (-self.width/2, self.width/2),
            (self.width/2, self.width/2),
            (3/2*self.width, 0),
            (self.width/2, -self.width/2),
        ]
        if init_position is not None:
            self.position = init_position
        else:
            self.position = np.array(self.trajectory[0][1])
        self.init_position = self.position.copy() 
        if init_heading is not None:
            self.heading = init_heading
        else:
            self.heading = np.pi/2

        if init_update:
            self.update(dt=0.1)

    def _update(self, dt):
        self.waypoint_counter += dt

        index = int(np.floor(self.waypoint_counter))

        if index >= len(self.trajectory_velocities) - 1:
            self.waypoint_counter = 0
            index = 0
            self.position = np.array(self.trajectory[0][1])

        dx = self.trajectory_velocities[index][0]
        dy = self.trajectory_velocities[index][1]

        self.dx = dt*dx
        self.dy = dt*dy
        self.heading = np.arctan2(self.dy, self.dx)
        self.position = self.position + np.array([self.dx, self.dy])
        self._prev_position.append(self.position)
        self._prev_heading.append(self.heading)

        return True

    def _calculate_boundary(self):
        ship_angle = self.heading# float(geom.princip(self.heading))

        boundary_temp = shapely.geometry.Polygon(self.points)
        boundary_temp = shapely.affinity.rotate(boundary_temp, ship_angle, use_radians=True, origin='centroid')
        boundary_temp = shapely.affinity.translate(boundary_temp, xoff=self.position[0], yoff=self.position[1])

        return boundary_temp


