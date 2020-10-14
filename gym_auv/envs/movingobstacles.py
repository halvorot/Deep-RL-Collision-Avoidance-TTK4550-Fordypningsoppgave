import numpy as np

import gym_auv.utils.geomutils as geom
import gym_auv.utils.helpers as helpers
from gym_auv.objects.vessel import Vessel
from gym_auv.objects.path import RandomCurveThroughOrigin, Path
from gym_auv.objects.obstacles import PolygonObstacle, VesselObstacle, CircularObstacle
from gym_auv.environment import BaseEnvironment
from gym_auv.objects.rewarder import ColavRewarder, ColregRewarder
import shapely.geometry, shapely.errors

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

vessel_speed_vals = np.loadtxt('resources/speed_vals.txt')
vessel_speed_density = np.loadtxt('resources/speed_density.txt')

class MovingObstacles(BaseEnvironment):

    def __init__(self, *args, **kwargs) -> None:
        """
        Sets following parameters for the scenario before calling super init. method:
            self._n_moving_obst : Number of moving obstacles
            self._n_static_obst : Number of static obstacles
            self._rewarder_class : Rewarder used, e.g. ColavRewarder, ColregRewarder
        """

        super().__init__(*args, **kwargs)

    def _generate(self):
        # Initializing path
        nwaypoints = int(np.floor(4*self.rng.rand() + 2))
        self.path = RandomCurveThroughOrigin(self.rng, nwaypoints, length=800)

        # Initializing vessel
        init_state = self.path(0)
        init_angle = self.path.get_direction(0)
        init_state[0] += 50*(self.rng.rand()-0.5)
        init_state[1] += 50*(self.rng.rand()-0.5)
        init_angle = geom.princip(init_angle + 2*np.pi*(self.rng.rand()-0.5))
        self.vessel = Vessel(self.config, np.hstack([init_state, init_angle]), width=self.config["vessel_width"])
        prog = 0
        self.path_prog_hist = np.array([prog])
        self.max_path_prog = prog
        
        self.obstacles = []

        # Adding moving obstacles
        for _ in range(self._n_moving_obst):
            other_vessel_trajectory = []

            obst_position, obst_radius = helpers.generate_obstacle(self.rng, self.path, self.vessel, obst_radius_mean=10, displacement_dist_std=500)
            obst_direction = self.rng.rand()*2*np.pi
            obst_speed = np.random.choice(vessel_speed_vals, p=vessel_speed_density)

            for i in range(10000):
                other_vessel_trajectory.append((i, (
                    obst_position[0] + i*obst_speed*np.cos(obst_direction), 
                    obst_position[1] + i*obst_speed*np.sin(obst_direction)
                )))
            other_vessel_obstacle = VesselObstacle(width=obst_radius, trajectory=other_vessel_trajectory)

            self.obstacles.append(other_vessel_obstacle)

        # Adding static obstacles
        for _ in range(self._n_static_obst):
            obstacle = CircularObstacle(*helpers.generate_obstacle(self.rng, self.path, self.vessel, displacement_dist_std=250))
            self.obstacles.append(obstacle)
        
        self._update()

class MovingObstaclesNoRules(MovingObstacles):
    def __init__(self, *args, **kwargs):
        self._n_moving_obst = 17
        self._n_static_obst = 11
        self._rewarder_class = ColavRewarder
        super().__init__(*args, **kwargs)

class MovingObstaclesColreg(MovingObstacles):
    def __init__(self, *args, **kwargs):
        self._n_moving_obst = 17
        self._n_static_obst = 11
        self._rewarder_class = ColregRewarder
        super().__init__(*args, **kwargs)