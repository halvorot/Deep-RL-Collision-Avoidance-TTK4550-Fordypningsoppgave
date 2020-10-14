import numpy as np
from abc import ABC, abstractmethod
from gym_auv.objects.vessel import Vessel

deg2rad = np.pi/180
rad2deg = 180/np.pi

def _sample_lambda(scale):
    log = -np.random.gamma(1, scale)
    y = np.power(10, log)
    return y

def _sample_eta():
    y = np.random.gamma(shape=1.9, scale=0.6)
    return y

class BaseRewarder(ABC):
    def __init__(self, vessel, test_mode) -> None:
        self._vessel = vessel
        self._test_mode = test_mode
        self.params = {}

    @property
    def vessel(self) -> Vessel:
        """Vessel instance that the reward is calculated with respect to."""
        return self._vessel[-1]

    @abstractmethod
    def calculate(self) -> float:
        """
        Calculates the step reward and decides whether the episode
        should be ended.

        Returns
        -------
        reward : float
            The reward for performing action at this timestep.
        """

    def insight(self) -> np.ndarray:
        """
        Returns a numpy array with reward parameters for the agent
        to have an insight into its reward function

        Returns
        -------
        insight : np.array
            The reward insight array at this timestep.
        """
        return np.array([])

class ColavRewarder(BaseRewarder):
    def __init__(self, vessel, test_mode):
        super().__init__(vessel, test_mode)
        self.params['gamma_theta'] = 10.0
        self.params['gamma_x'] = 0.1
        self.params['gamma_v_y'] = 1.0
        self.params['gamma_y_e'] = 5.0
        self.params['penalty_yawrate'] = 0.0
        self.params['penalty_torque_change'] = 0.0
        self.params['cruise_speed'] = 0.1
        self.params['neutral_speed'] = 0.05
        self.params['negative_multiplier'] = 2.0
        self.params['collision'] = -10000.0
        self.params['lambda'] = 0.5 #_sample_lambda(scale=0.2)
        self.params['eta'] = 0#_sample_eta()

    N_INSIGHTS = 0
    def insight(self):
        return np.array([])
        #return np.array([np.log10(self.params['lambda'])])
    
    def calculate(self):
        latest_data = self._vessel.req_latest_data()
        nav_states = latest_data['navigation']
        measured_distances = latest_data['distance_measurements']
        measured_speeds = latest_data['speed_measurements']
        collision = latest_data['collision']

        if collision:
            reward = self.params["collision"]*(1-self.params["lambda"])
            return reward
         
        reward = 0

        # Extracting navigation states
        cross_track_error = nav_states['cross_track_error']
        heading_error = nav_states['heading_error']

        # Calculating path following reward component
        cross_track_performance = np.exp(-self.params['gamma_y_e']*np.abs(cross_track_error))
        path_reward = (1 + np.cos(heading_error)*self._vessel.speed/self._vessel.max_speed)*(1 + cross_track_performance) - 1
        
        # Calculating obstacle avoidance reward component
        closeness_penalty_num = 0
        closeness_penalty_den = 0
        if self._vessel.n_sensors > 0:
            for isensor in range(self._vessel.n_sensors):
                angle = self._vessel.sensor_angles[isensor]
                x = measured_distances[isensor]
                speed_vec = measured_speeds[isensor]
                weight = 1 / (1 + np.abs(self.params['gamma_theta']*angle))
                raw_penalty = self._vessel.config["sensor_range"]*np.exp(-self.params['gamma_x']*x +self.params['gamma_v_y']*max(0, speed_vec[1]))
                weighted_penalty = weight*raw_penalty
                closeness_penalty_num += weighted_penalty
                closeness_penalty_den += weight

            closeness_reward = -closeness_penalty_num/closeness_penalty_den
        else:
            closeness_reward = 0

        # Calculating living penalty
        living_penalty = self.params['lambda']*(2*self.params["neutral_speed"]+1) + self.params["eta"]*self.params["neutral_speed"]
        
        # Calculating total reward
        reward = self.params['lambda']*path_reward + \
            (1-self.params['lambda'])*closeness_reward - \
            living_penalty + \
            self.params["eta"]*self._vessel.speed/self._vessel.max_speed - \
            self.params["penalty_yawrate"]*abs(self._vessel.yaw_rate)

        if reward < 0:
            reward *= self.params['negative_multiplier']

        return reward

class ColregRewarder(BaseRewarder):
    def __init__(self, vessel, test_mode):
        super().__init__(vessel, test_mode)
        self.params['gamma_theta'] = 10.0
        self.params['gamma_x_stat'] = 0.09
        self.params['gamma_x_starboard'] = 0.07
        self.params['gamma_x_port'] = 0.09
        self.params['gamma_v_y'] = 2.0
        self.params['gamma_y_e'] = 5.0
        self.params['penalty_yawrate'] = 0.0
        self.params['penalty_torque_change'] = 0.01
        self.params['cruise_speed'] = 0.1
        self.params['neutral_speed'] = 0.1
        self.params['negative_multiplier'] = 2.0
        self.params['collision'] = -10000.0
        self.params['lambda'] = 0.5#_sample_lambda(scale=0.2)
        self.params['eta'] = 0.2#_sample_eta()
        self.params['alpha_lambda'] = 3.5
        self.params['gamma_min_x'] = 0.04
        self.params['gamma_weight'] = 2


    N_INSIGHTS = 1
    def insight(self):
        return np.array ([self.params['lambda']])

    def calculate(self):
        latest_data = self._vessel.req_latest_data()
        nav_states = latest_data['navigation']
        measured_distances = latest_data['distance_measurements']
        measured_speeds = latest_data['speed_measurements']
        collision = latest_data['collision']
        #print([x[1] for x in measured_speeds])
        if collision:
            reward = self.params['collision']#*(1-self.params['lambda'])
            return reward

        reward = 0

        # Extracting navigation states
        cross_track_error = nav_states['cross_track_error']
        heading_error = nav_states['heading_error']

        # Calculating path following reward component
        cross_track_performance = np.exp(-self.params['gamma_y_e']*np.abs(cross_track_error))
        path_reward = (1 + np.cos(heading_error)*self._vessel.speed/self._vessel.max_speed)*(1 + cross_track_performance) - 1

        # Calculating obstacle avoidance reward component
        closeness_penalty_num = 0
        closeness_penalty_den = 0
        static_closeness_penalty_num = 0
        static_closeness_penalty_den = 0
        closeness_reward = 0
        static_closeness_reward = 0
        moving_distances = []
        lambdas = []

        speed_weight = 2

        if self._vessel.n_sensors > 0:
            for isensor in range(self._vessel.n_sensors):
                angle = self._vessel.sensor_angles[isensor]
                x = measured_distances[isensor]
                speed_vec = measured_speeds[isensor]


                if speed_vec.any():

                    if speed_vec[1] > 0:
                        self.params['lambda'] = 1/(1+np.exp(-0.04*x+4))
                    if speed_vec[1] < 0:
                        self.params['lambda'] = 1/(1+np.exp(-0.06*x+3))
                    lambdas.append(self.params['lambda'])

                    weight = 2 / (1 + np.exp(self.params['gamma_weight']*np.abs(angle)))
                    moving_distances.append(x)

                    if angle < 0*deg2rad and angle > -112.5*deg2rad: #straffer høyre side

                        raw_penalty = 100*np.exp(-self.params['gamma_x_starboard']*x + speed_weight*speed_vec[1])
                    else:
                        raw_penalty = 100*np.exp(-self.params['gamma_x_port']*x + speed_weight*speed_vec[1])
                    #else:
                    #    raw_penalty = 100*np.exp(-self.params['gamma_x_stat']*x)

                    weighted_penalty = (1-self.params['lambda'])*weight*raw_penalty
                    closeness_penalty_num += weighted_penalty
                    closeness_penalty_den += weight

                else:

                    weight = 1 / (1 + np.abs(self.params['gamma_theta']*angle))
                    raw_penalty = 100*np.exp(-self.params['gamma_x_stat']*x)
                    weighted_penalty = weight*raw_penalty
                    static_closeness_penalty_num += weighted_penalty
                    static_closeness_penalty_den += weight



            if closeness_penalty_num:
                closeness_reward = -closeness_penalty_num/closeness_penalty_den


            if static_closeness_penalty_num:
                static_closeness_reward = -static_closeness_penalty_num/static_closeness_penalty_den


        #if len(moving_distances) != 0:
        #    min_dist = np.amin(moving_distances)
        #    self.params['lambda'] = 1/(1+np.exp(-0.04*min_dist+4))
            #self.params['lambda'] = 1/(1+np.exp(-(self.params['gamma_min_x']*min_dist-self.params['alpha_lambda'])))
        #else:
        #    self.params['lambda'] = 1

        if len(lambdas):
            path_lambda = np.amin(lambdas)
        else:
            path_lambda = 1

        #if path_reward > 0:
        #    path_reward = path_lambda*path_reward
        # Calculating living penalty
        living_penalty = 1#.2*(self.params['neutral_speed']+1) + self.params['eta']*self.params['neutral_speed']

        # Calculating total reward
        reward = path_lambda*path_reward + \
            static_closeness_reward + \
            closeness_reward - \
            living_penalty + \
            self.params['eta']*self._vessel.speed/self._vessel.max_speed# - \
            #self.params['penalty_yawrate']*abs(self._vessel.yaw_rate)

        if reward < 0:
            reward *= self.params['negative_multiplier']

        return reward