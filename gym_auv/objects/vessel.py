"""
This module implements an AUV that is simulated in the horizontal plane.
"""
import numpy as np
import numpy.linalg as linalg
from itertools import islice, chain, repeat
import shapely.geometry, shapely.errors, shapely.strtree, shapely.ops, shapely.prepared

import gym_auv.utils.constants as const
import gym_auv.utils.geomutils as geom
from gym_auv.objects.obstacles import LineObstacle
from gym_auv.objects.path import Path

def _odesolver45(f, y, h):
    """Calculate the next step of an IVP of a time-invariant ODE with a RHS
    described by f, with an order 4 approx. and an order 5 approx.
    Parameters:
        f: function. RHS of ODE.
        y: float. Current position.
        h: float. Step length.
    Returns:
        q: float. Order 2 approx.
        w: float. Order 3 approx.
    """
    s1 = f(y)
    s2 = f(y+h*s1/4.0)
    s3 = f(y+3.0*h*s1/32.0+9.0*h*s2/32.0)
    s4 = f(y+1932.0*h*s1/2197.0-7200.0*h*s2/2197.0+7296.0*h*s3/2197.0)
    s5 = f(y+439.0*h*s1/216.0-8.0*h*s2+3680.0*h*s3/513.0-845.0*h*s4/4104.0)
    s6 = f(y-8.0*h*s1/27.0+2*h*s2-3544.0*h*s3/2565+1859.0*h*s4/4104.0-11.0*h*s5/40.0)
    w = y + h*(25.0*s1/216.0+1408.0*s3/2565.0+2197.0*s4/4104.0-s5/5.0)
    q = y + h*(16.0*s1/135.0+6656.0*s3/12825.0+28561.0*s4/56430.0-9.0*s5/50.0+2.0*s6/55.0)
    return w, q

def _standardize_intersect(intersect):
    if intersect.is_empty:
        return []
    elif isinstance(intersect, shapely.geometry.LineString):
        return [shapely.geometry.Point(intersect.coords[0])]
    elif isinstance(intersect, shapely.geometry.Point):
        return [intersect]
    else:
        return list(intersect.geoms)

def _feasibility_pooling(x, width, theta):
    N_sensors = x.shape[0]
    sort_idx = np.argsort(x)
    for idx in sort_idx:
        surviving = x > x[idx] + width
        d = x[idx]*theta
        opening_width = 0
        opening_span = 0
        opening_start = -theta*(N_sensors-1)/2
        found_opening = False
        for isensor, sensor_survives in enumerate(surviving):
            if sensor_survives:
                opening_width += d
                opening_span += theta
                if opening_width > width:
                    opening_center = opening_start + opening_span/2
                    if abs(opening_center) < theta*(N_sensors-1)/4:
                        found_opening = True
            else:
                opening_width += 0.5*d
                opening_span += 0.5*theta
                if opening_width > width:
                    opening_center = opening_start + opening_span/2
                    if abs(opening_center) < theta*(N_sensors-1)/4:
                        found_opening = True
                opening_width = 0
                opening_span = 0
                opening_start = -theta*(N_sensors-1)/2 + isensor*theta

        if not found_opening: 
            return max(0, x[idx])

    return max(0, np.max(x))

def _simulate_sensor(sensor_angle, p0_point, sensor_range, obstacles):
    sensor_endpoint = (
        p0_point.x + np.cos(sensor_angle)*sensor_range,
        p0_point.y + np.sin(sensor_angle)*sensor_range
    )
    sector_ray = shapely.geometry.LineString([p0_point, sensor_endpoint])

    obst_intersections = [sector_ray.intersection(elm.boundary) for elm in obstacles]
    obst_intersections = list(map(_standardize_intersect, obst_intersections))
    obst_references = list(chain.from_iterable(repeat(obstacles[i], len(obst_intersections[i])) for i in range(len(obst_intersections))))
    obst_intersections = list(chain(*obst_intersections))

    if obst_intersections:
        measured_distance, intercept_idx = min((float(p0_point.distance(elm)), i) for i, elm in enumerate(obst_intersections))
        obstacle = obst_references[intercept_idx]
        if not obstacle.static:
            obst_speed_homogenous = geom.to_homogeneous([obstacle.dx, obstacle.dy])
            obst_speed_rel_homogenous = geom.Rz(-sensor_angle - np.pi/2).dot(obst_speed_homogenous)
            obst_speed_vec_rel = geom.to_cartesian(obst_speed_rel_homogenous)
        else:
            obst_speed_vec_rel = (0, 0)
        ray_blocked = True
    else:
        measured_distance = sensor_range
        obst_speed_vec_rel = (0, 0)
        ray_blocked = False

    return (measured_distance, obst_speed_vec_rel, ray_blocked)

class Vessel():

    NAVIGATION_FEATURES = [
        'surge_velocity',
        'sway_velocity',
        'yaw_rate',
        'look_ahead_heading_error',
        'heading_error',
        'cross_track_error'
    ]

    def __init__(self, config:dict, init_state:np.ndarray, width:float=4) -> None:
        """
        Initializes and resets the vessel.

        Parameters
        ----------
        config : dict
            Dictionary containing the configuration parameters for
            the vessel
        init_state : np.ndarray
            The initial attitude of the veHssel [x, y, psi], where
            psi is the initial heading of the AUV.
        width : float
            The distance from the center of the AUV to its edge
            in meters.
        """
        
        self.config = config

        # Initializing private attributes
        self._width = width
        self._feasibility_width = width*self.config["feasibility_width_multiplier"]
        self._n_sectors = self.config["n_sectors"]
        self._n_sensors = self.config["n_sensors_per_sector"]*self.config["n_sectors"]
        self._d_sensor_angle = 2*np.pi/(self._n_sensors)
        self._sensor_angles = np.array([-np.pi + (i + 1)*self._d_sensor_angle for i in range(self._n_sensors)])
        self._sector_angles = []
        self._n_sensors_per_sector = [0]*self._n_sectors
        self._sector_start_indeces = [0]*self._n_sectors
        self._sector_end_indeces = [0]*self._n_sectors
        self._sensor_internal_indeces = []
        self._sensor_interval = max(1, int(1/self.config["sensor_frequency"]))
        self._observe_interval = max(1, int(1/self.config["observe_frequency"]))
        self._virtual_environment = None

        # Calculating sensor partitioning
        last_isector = -1
        tmp_sector_angle_sum = 0
        tmp_sector_sensor_count = 0
        for isensor in range(self._n_sensors):
            isector = self.config["sector_partition_fun"](self, isensor)
            angle = self._sensor_angles[isensor]
            if isector == last_isector:
                tmp_sector_angle_sum += angle
                tmp_sector_sensor_count += 1
            else:
                if last_isector > -1:
                    self._sector_angles.append(tmp_sector_angle_sum/tmp_sector_sensor_count)
                last_isector = isector
                self._sector_start_indeces[isector] = isensor
                tmp_sector_angle_sum = angle
                tmp_sector_sensor_count = 1
            self._n_sensors_per_sector[isector] += 1
        self._sector_angles.append(tmp_sector_angle_sum/tmp_sector_sensor_count)
        self._sector_angles = np.array(self._sector_angles)

        for isensor in range(self._n_sensors):
            isector = self.config["sector_partition_fun"](self, isensor)
            isensor_internal = isensor - self._sector_start_indeces[isector]
            self._sensor_internal_indeces.append(isensor_internal)

        for isector in range(self._n_sectors):
            self._sector_end_indeces[isector] = self._sector_start_indeces[isector] + self._n_sensors_per_sector[isector]

        # Calculating feasible closeness
        if self.config["sensor_log_transform"]:
            self._get_closeness = lambda x: 1 - np.clip(np.log(1 + x)/np.log(1 + self.config["sensor_range"]), 0, 1)
        else:
            self._get_closeness = lambda x: 1 - np.clip(x/self.config["sensor_range"], 0, 1)

        # Initializing vessel to initial position
        self.reset(init_state)

    @property
    def n_sensors(self) -> int:
        """Number of sensors."""
        return self._n_sensors

    @property
    def width(self) -> float:
        """Width of vessel in meters."""
        return self._width

    @property
    def position(self) -> np.ndarray:
        """Returns an array holding the position of the AUV in cartesian
        coordinates."""
        return self._state[0:2]

    @property
    def path_taken(self) -> np.ndarray:
        """Returns an array holding the path of the AUV in cartesian
        coordinates."""
        return self._prev_states[:, 0:2]
    
    @property
    def heading_taken(self) -> np.ndarray:
        """Returns an array holding the heading of the AUV for all timesteps."""
        return self._prev_states[:, 2]

    @property
    def heading(self) -> float:
        """Returns the heading of the AUV with respect to true north."""
        return self._state[2]

    @property
    def velocity(self) -> np.ndarray:
        """Returns the surge and sway velocity of the AUV."""
        return self._state[3:5]

    @property
    def speed(self) -> float:
        """Returns the speed of the AUV."""
        return linalg.norm(self.velocity)

    @property
    def yaw_rate(self) -> float:
        """Returns the rate of rotation about the z-axis."""
        return self._state[5]

    @property
    def max_speed(self) -> float:
        """Returns the maximum speed of the AUV."""
        return 2

    @property
    def course(self) -> float:
        """Returns the course angle of the AUV with respect to true north."""
        crab_angle = np.arctan2(self.velocity[1], self.velocity[0])
        return self.heading + crab_angle

    @property
    def sensor_angles(self) -> np.ndarray:
        """Array containg the angles each sensor ray relative to the vessel heading."""
        return self._sensor_angles

    @property
    def sector_angles(self) -> np.ndarray:
        """Array containg the angles of the center line of each sensor sector relative to the vessel heading."""
        return self._sector_angles

    def reset(self, init_state:np.ndarray) -> None:
        """
        Resets the vessel to the specified initial state.

        Parameters
        ----------
        init_state : np.ndarray
            The initial attitude of the veHssel [x, y, psi], where
            psi is the initial heading of the AUV.
        """
        init_speed = [0, 0, 0]
        init_state = np.array(init_state, dtype=np.float64)
        init_speed = np.array(init_speed, dtype=np.float64)
        self._state = np.hstack([init_state, init_speed])
        self._prev_states = np.vstack([self._state])
        self._input = [0, 0]
        self._prev_inputs =np.vstack([self._input])
        self._last_sensor_dist_measurements = np.ones((self._n_sensors,))*self.config["sensor_range"]
        self._last_sensor_speed_measurements = np.zeros((self._n_sensors,2))
        self._last_sector_dist_measurements = np.zeros((self._n_sectors,))
        self._last_sector_feasible_dists = np.zeros((self._n_sectors,))
        self._last_navi_state_dict = dict((state, 0) for state in Vessel.NAVIGATION_FEATURES)
        self._virtual_environment = None
        self._collision = False
        self._progress = 0
        self._reached_goal = False

        self._step_counter = 0
        self._perceive_counter = 0
        self._nearby_obstacles = []

    def step(self, action:list) -> None:
        """
        Simulates the vessel one step forward after applying the given action.

        Parameters
        ----------
        action : np.ndarray[thrust_input, torque_input]
        """
        self._input = np.array([self._thrust_surge(action[0]), self._moment_steer(action[1])])
        w, q = _odesolver45(self._state_dot, self._state, self.config["t_step_size"])
        
        self._state = q
        self._state[2] = geom.princip(self._state[2])

        self._prev_states = np.vstack([self._prev_states,self._state])
        self._prev_inputs = np.vstack([self._prev_inputs,self._input])

        self._step_counter += 1

    def perceive(self, obstacles:list) -> (np.ndarray, np.ndarray):
        """
        Simulates the sensor suite and returns observation arrays of the environment.
        
        Returns
        -------
        sector_closenesses : np.ndarray
        sector_velocities : np.ndarray
        """

        # Initializing variables
        sensor_range = self.config["sensor_range"]
        p0_point = shapely.geometry.Point(*self.position)

        # Loading nearby obstacles, i.e. obstacles within the vessel's detection range
        if self._step_counter % self.config["sensor_interval_load_obstacles"] == 0:
            self._nearby_obstacles = list(filter(
                lambda obst: float(p0_point.distance(obst.boundary)) - self._width < sensor_range, obstacles
            ))

        if not self._nearby_obstacles:
            self._last_sensor_dist_measurements = np.ones((self._n_sensors,))*sensor_range
            sector_feasible_distances = np.ones((self._n_sectors,))*sensor_range
            sector_closenesses = np.zeros((self._n_sectors,))
            sector_velocities = np.zeros((2*self._n_sectors,))
            collision = False

        else:
            should_observe = (self._perceive_counter % self._observe_interval == 0) or self._virtual_environment is None
            if should_observe:
                geom_targets = self._nearby_obstacles
            else:
                geom_targets = self._virtual_environment

            # Simulating all sensors using _simulate_sensor subroutine
            sensor_angles_ned = self._sensor_angles + self.heading
            activate_sensor = lambda i: (i % self._sensor_interval) == (self._perceive_counter % self._sensor_interval)
            sensor_sim_args = (p0_point, sensor_range, geom_targets)
            sensor_output_arrs = list(map(
                lambda i: _simulate_sensor(sensor_angles_ned[i], *sensor_sim_args) if activate_sensor(i) else (
                    self._last_sensor_dist_measurements[i],
                    self._last_sensor_speed_measurements[i],
                    True
                ), 
                range(self._n_sensors)
            ))
            sensor_dist_measurements, sensor_speed_measurements, sensor_blocked_arr = zip(*sensor_output_arrs)
            sensor_dist_measurements = np.array(sensor_dist_measurements)
            sensor_speed_measurements = np.array(sensor_speed_measurements)
            self._last_sensor_dist_measurements = sensor_dist_measurements
            self._last_sensor_speed_measurements = sensor_speed_measurements

            # Setting virtual obstacle
            if should_observe:
                line_segments = []
                tmp = []
                for i in range(self.n_sensors):
                    if sensor_blocked_arr[i]:
                        point = (
                            self.position[0] + np.cos(sensor_angles_ned[i])*sensor_dist_measurements[i],
                            self.position[1] + np.sin(sensor_angles_ned[i])*sensor_dist_measurements[i]
                        )
                        tmp.append(point)
                    elif len(tmp) > 1:
                        line_segments.append(tuple(tmp))
                        tmp = []

                self._virtual_environment = list(map(LineObstacle, line_segments))

            # Partitioning sensor readings into sectors
            sector_dist_measurements = np.split(sensor_dist_measurements, self._sector_start_indeces[1:])
            sector_speed_measurements = np.split(sensor_speed_measurements, self._sector_start_indeces[1:], axis=0)

            # Performing feasibility pooling
            sector_feasible_distances = np.array(list(
                map(lambda x: _feasibility_pooling(x, self._feasibility_width, self._d_sensor_angle), sector_dist_measurements)
            ))

            # Calculating feasible closeness
            sector_closenesses = self._get_closeness(sector_feasible_distances)

            # Retrieving obstacle speed for closest obstacle within each sector
            closest_obst_sensor_indeces = list(map(np.argmin, sector_dist_measurements))
            sector_velocities = np.concatenate(
                [sector_speed_measurements[i][closest_obst_sensor_indeces[i]] for i in range(self._n_sectors)]
            )

            # Testing if vessel has collided
            collision = np.any(sensor_dist_measurements < self.width)

        self._last_sector_dist_measurements = sector_closenesses
        self._last_sector_feasible_dists = sector_feasible_distances
        self._collision = collision
        self._perceive_counter += 1

        return (sector_closenesses, sector_velocities)

    def navigate(self, path:Path) -> np.ndarray:
        """
        Calculates and returns navigation states representing the vessel's attitude
        with respect to the desired path.
        
        Returns
        -------
        navigation_states : np.ndarray
        """

        # Calculating path arclength at reference point, i.e. the point closest to the vessel
        vessel_arclength = path.get_closest_arclength(self.position)

        # Calculating tangential path direction at reference point
        path_direction = path.get_direction(vessel_arclength)
        cross_track_error = geom.Rzyx(0, 0, -path_direction).dot(
            np.hstack([path(vessel_arclength) - self.position, 0])
        )[1]

        # Calculating tangential path direction at look-ahead point
        target_arclength = min(path.length, vessel_arclength + self.config["look_ahead_distance"])
        look_ahead_path_direction = path.get_direction(target_arclength) 
        look_ahead_heading_error = float(geom.princip(look_ahead_path_direction - self.heading))

        # Calculating vector difference between look-ahead point and vessel position
        target_vector = path(target_arclength) - self.position

        # Calculating heading error
        target_heading = np.arctan2(target_vector[1], target_vector[0])
        heading_error = float(geom.princip(target_heading - self.heading))

        # Calculating path progress
        progress = vessel_arclength/path.length
        self._progress = progress

        # Deciding if vessel has reached the goal
        goal_distance = linalg.norm(path.end - self.position)
        reached_goal = goal_distance <= self.config["min_goal_distance"] or progress >= self.config["min_path_progress"]
        self._reached_goal = reached_goal

        # Concatenating states
        self._last_navi_state_dict = {
            'surge_velocity': self.velocity[0],
            'sway_velocity': self.velocity[1],
            'yaw_rate': self.yaw_rate,
            'look_ahead_heading_error': look_ahead_heading_error,
            'heading_error': heading_error,
            'cross_track_error': cross_track_error/100,
            'target_heading': target_heading,
            'look_ahead_path_direction': look_ahead_path_direction,
            'path_direction': path_direction,
            'vessel_arclength': vessel_arclength,
            'target_arclength': target_arclength,
            'goal_distance': goal_distance
        }
        navigation_states = np.array([self._last_navi_state_dict[state] for state in Vessel.NAVIGATION_FEATURES])

        return navigation_states

    def req_latest_data(self) -> dict:
        """Returns dictionary containing the most recent perception and navigation
        states."""
        return {
            'distance_measurements': self._last_sensor_dist_measurements,
            'speed_measurements': self._last_sensor_speed_measurements,
            'feasible_distances': self._last_sector_feasible_dists,
            'navigation': self._last_navi_state_dict,
            'collision' : self._collision,
            'progress': self._progress,
            'reached_goal': self._reached_goal
        }

    def _state_dot(self, state):
        psi = state[2]
        nu = state[3:]

        tau = np.array([self._input[0], 0, self._input[1]])

        eta_dot = geom.Rzyx(0, 0, geom.princip(psi)).dot(nu)
        nu_dot = const.M_inv.dot(
            tau
            #- const.D.dot(nu)
            - const.N(nu).dot(nu)
        )
        state_dot = np.concatenate([eta_dot, nu_dot])
        return state_dot

    def _thrust_surge(self, surge):
        surge = np.clip(surge, 0, 1)
        return surge*self.config['thrust_max_auv']

    def _moment_steer(self, steer):
        steer = np.clip(steer, -1, 1)
        return steer*self.config['moment_max_auv']