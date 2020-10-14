from copy import deepcopy
import numpy as np
import numpy.linalg as linalg
import shapely.geometry
from scipy.optimize import minimize

from scipy import interpolate

import gym_auv.utils.geomutils as geom

def _arc_len(coords):
    diff = np.diff(coords, axis=1)
    delta_arc = np.sqrt(np.sum(diff ** 2, axis=0))
    return np.concatenate([[0], np.cumsum(delta_arc)])

class Path():
    def __init__(self, waypoints:list) -> None:
        """Initializes path based on specified waypoints."""

        self.init_waypoints = waypoints.copy()

        for _ in range(3):
            self._arclengths = _arc_len(waypoints)
            path_coords = interpolate.pchip(x=self._arclengths, y=waypoints, axis=1)
            path_derivatives = path_coords.derivative()
            path_dderivatives = path_derivatives.derivative()
            waypoints = path_coords(np.linspace(self._arclengths[0], self._arclengths[-1], 1000))

        self._waypoints = waypoints.copy()
        self._path_coords = path_coords
        self._path_derivatives = path_derivatives
        self._path_dderivatives = path_dderivatives

        S = np.linspace(0, self.length, 10*self.length)
        self._points = np.transpose(self._path_coords(S))
        self._linestring = shapely.geometry.LineString(self._points)

    @property
    def length(self) -> float:
        """Length of path in meters."""
        return self._arclengths[-1]

    @property
    def start(self) -> np.ndarray:
        """Coordinates of the path's starting point."""
        return self._path_coords(0)

    @property
    def end(self) -> np.ndarray:
        """Coordinates of the path's end point."""
        return self._path_coords(self.length)

    def __call__(self, arclength:float) -> np.ndarray:
        """
        Returns the (x,y) point corresponding to the
        specified arclength.
        
        Returns
        -------
        point : np.array
        """
        return self._path_coords(arclength)

    def get_direction(self, arclength:float) -> float:
        """
        Returns the direction in radians with respect to the
        positive x-axis.
        
        Returns
        -------
        direction : float
        """
        derivative = self._path_derivatives(arclength)
        return np.arctan2(derivative[1], derivative[0])

    def get_closest_arclength(self, position:np.ndarray) -> float:  
        """
        Returns the arc length value corresponding to the point 
        on the path which is closest to the specified position.
        
        Returns
        -------
        point : np.array
        """
        return self._linestring.project(shapely.geometry.Point(position))

class RandomCurveThroughOrigin(Path):
    def __init__(self, rng, nwaypoints, length=400):
        angle_init = 2*np.pi*(rng.rand() - 0.5)
        start = np.array([0.5*length*np.cos(angle_init), 0.5*length*np.sin(angle_init)])
        end = -np.array(start)
        waypoints = np.vstack([start, end])
        for waypoint in range(nwaypoints // 2):
            newpoint1 = ((nwaypoints // 2 - waypoint)
                         * start / (nwaypoints // 2 + 1)
                         + length / (nwaypoints // 2 + 1)
                         * (rng.rand()-0.5))
            newpoint2 = ((nwaypoints // 2 - waypoint)
                         * end / (nwaypoints // 2 + 1)
                         + length / (nwaypoints // 2 + 1)
                         * (rng.rand()-0.5))
            waypoints = np.vstack([waypoints[:waypoint+1, :],
                                   newpoint1,
                                   np.array([0, 0]),
                                   newpoint2,
                                   waypoints[-1*waypoint-1:, :]])
        super().__init__(np.transpose(waypoints))