import gym
import numpy as np
from gym.utils import seeding

from gym_auv.objects.vessel import Vessel
from gym_auv.objects.rewarder import ColavRewarder
import gym_auv.rendering.render2d as render2d
import gym_auv.rendering.render3d as render3d
from abc import ABC, abstractmethod

class BaseEnvironment(gym.Env, ABC):
    """Creates an environment with a vessel and a path."""

    metadata = {
        'render.modes': ['human', 'rgb_array', 'state_pixels'],
        'video.frames_per_second': render2d.FPS
    }

    def __init__(self, env_config, test_mode=False, render_mode='2d', verbose=False):
        """The __init__ method declares all class atributes and calls
        the self.reset() to intialize them properly.

        Parameters
        ----------
            env_config : dict
                Configuration parameters for the environment. 
                The default values are set in __init__.py
            test_mode : bool
                If test_mode is True, the environment will not be autonatically reset 
                due to too low cumulative reward or too large distance from the path. 
            render_mode : {'2d', '3d', 'both'}
                Whether to use 2d or 3d rendering. 'both' is currently broken.
            verbose
                Whether to print debugging information.
        """

        if not hasattr(self, '_rewarder_class'):
            self._rewarder_class = ColavRewarder
            self._n_moving_obst = 10
            self._n_moving_stat = 10
        
        self.test_mode = test_mode
        self.render_mode = render_mode
        self.verbose = verbose
        self.config = env_config
        
        # Setting dimension of observation vector
        self.n_observations = len(Vessel.NAVIGATION_FEATURES) + 3*self.config["n_sectors"] + self._rewarder_class.N_INSIGHTS

        self.episode = 0
        self.total_t_steps = 0
        self.t_step = 0
        self.cumulative_reward = 0
        self.history = []
        self.rewarder = None

        # Declaring attributes
        self.obstacles = []
        self.vessel = None
        self.path = None
        
        self.reached_goal = None
        self.collision = None
        self.progress = None
        self.last_reward = None
        self.last_episode = None
        self.rng = None
        self.seed()
        self._tmp_storage = None
        self._last_image_frame = None

        self._action_space = gym.spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            dtype=np.float32
        )
        self._observation_space = gym.spaces.Box(
            low=np.array([-1]*self.n_observations),
            high=np.array([1]*self.n_observations),
            dtype=np.float32
        )

        # Initializing rendering
        self._viewer2d = None
        self._viewer3d = None
        if self.render_mode == '2d' or self.render_mode == 'both':
            render2d.init_env_viewer(self)
        if self.render_mode == '3d' or self.render_mode == 'both':
            if self.config['render_distance'] == 'random':
                self.render_distance = self.rng.randint(300, 2000)
            else:
                self.render_distance = self.config['render_distance']
            render3d.init_env_viewer(self, autocamera=self.config["autocamera3d"], render_dist=self.render_distance)

        self.reset()

    @property
    def action_space(self) -> gym.spaces.Box:
        """Array defining the shape and bounds of the agent's action."""
        return self._action_space

    @property
    def observation_space(self) -> gym.spaces.Box:
        """Array defining the shape and bounds of the agent's observations."""
        return self._observation_space

    def reset(self, save_history=True):
        """Reset the environment's state. Returns observation.

        Returns
        -------
        obs : np.ndarray
            The initial observation of the environment.
        """

        if self.verbose: print('Resetting environment... Last reward was {:.2f}'.format(self.cumulative_reward))

        # Seeding
        if self.rng is None:
            self.seed()

        # Saving information about episode
        if self.t_step:
           self.save_latest_episode(save_history=save_history)

        # Incrementing counters
        self.episode += 1
        self.total_t_steps += self.t_step

        # Resetting all internal variables
        self.cumulative_reward = 0
        self.t_step = 0
        self.last_reward = 0
        self.reached_goal = False
        self.collision = False
        self.progress = 0
        self._last_image_frame = None

        # Generating a new environment
        if self.verbose:    print('Generating scenario...')
        self._generate()
        self.rewarder = self._rewarder_class(self.vessel, self.test_mode) # Resetting rewarder instance
        if self.verbose:    print('Generated scenario')

        # Initializing 3d viewer
        if self.render_mode == '3d':
            render3d.init_boat_model(self)
            #self._viewer3d.create_path(self.path)

        # Getting initial observation vector
        obs = self.observe()
        if self.verbose:    print('Calculated initial observation')

        # Resetting temporary data storage
        self._tmp_storage = {
            'cross_track_error': [],
        }

        return obs

    def observe(self) -> np.ndarray:
        """Returns the array of observations at the current time-step.

        Returns
        -------
        obs : np.ndarray
            The observation of the environment.
        """
        reward_insight = self.rewarder.insight()
        navigation_states = self.vessel.navigate(self.path)
        if bool(self.config["sensing"]):
            sector_closenesses, sector_velocities = self.vessel.perceive(self.obstacles)
        else:
            sector_closenesses, sector_velocities = [], []

        obs = np.concatenate([reward_insight, navigation_states, sector_closenesses, sector_velocities])
        return obs

    def step(self, action:list) -> (np.ndarray, float, bool, dict):
        """
        Steps the environment by one timestep. Returns observation, reward, done, info.

        Parameters
        ----------
        action : np.ndarray
            [thrust_input, torque_input].

        Returns
        -------
        obs : np.ndarray
            Observation of the environment after action is performed.
        reward : double
            The reward for performing action at his timestep.
        done : bool
            If True the episode is ended, due to either a collision or having reached the goal position.
        info : dict
            Dictionary with data used for reporting or debugging
        """

        action[0] = (action[0] + 1)/2 # Done to be compatible with RL algorithms that require symmetric action spaces
        if np.isnan(action).any(): action = np.zeros(action.shape)

        # If the environment is dynamic, calling self.update will change it.
        self._update()

        # Updating vessel state from its dynamics model
        self.vessel.step(action)

        # Getting observation vector
        obs = self.observe()
        vessel_data = self.vessel.req_latest_data()
        self.collision = vessel_data['collision']
        self.reached_goal = vessel_data['reached_goal']
        self.goal_distance = vessel_data['navigation']['goal_distance']
        self.progress = vessel_data['progress']

        # Receiving agent's reward
        reward = self.rewarder.calculate()
        self.last_reward = reward
        self.cumulative_reward += reward

        info = {}
        info['collision'] = self.collision
        info['reached_goal'] = self.reached_goal
        info['goal_distance'] = self.goal_distance
        info['progress'] = self.progress

        # Testing criteria for ending the episode
        done = self._isdone()

        self._save_latest_step()

        self.t_step += 1

        return (obs, reward, done, info)

    def _isdone(self) -> bool:
        return any([
            self.collision,
            self.reached_goal,
            self.t_step > self.config["max_timesteps"] and not self.test_mode,
            self.cumulative_reward < self.config["min_cumulative_reward"] and not self.test_mode
        ])

    def _update(self) -> None:
        """Updates the environment at each time-step. Can be customized in sub-classes."""
        [obst.update(dt=self.config["t_step_size"]) for obst in self.obstacles if not obst.static]

    @abstractmethod
    def _generate(self) -> None:    
        """Create new, stochastically genereated scenario. 
        To be implemented in extensions of BaseEnvironment. Must set the
        'vessel', 'path' and 'obstacles' attributes.
        """

    def close(self):
        """Closes the environment. To be called after usage."""
        if self._viewer2d is not None:
            self._viewer2d.close()
        if self._viewer3d is not None:
            self._viewer3d.close()

    def render(self, mode='human'):
        """Render one frame of the environment. 
        The default mode will do something human friendly, such as pop up a window."""
        image_arr = None
        try:
            if self.render_mode == '2d' or self.render_mode == 'both':
                image_arr = render2d.render_env(self, mode)
            if self.render_mode == '3d' or self.render_mode == 'both':
                image_arr = render3d.render_env(self, mode, self.config["t_step_size"])
        except OSError:
            image_arr = self._last_image_frame

        if image_arr is None:
            image_arr = self._last_image_frame
        else:
            self._last_image_frame = image_arr

        if image_arr is None and mode == 'rgb_array':
            print('Warning: image_arr is None -> video is likely broken' )

        return image_arr

    def seed(self, seed=None):
        """Reseeds the random number generator used in the environment"""
        self.rng, seed = seeding.np_random(seed)
        return [seed]

    def _save_latest_step(self):
        latest_data = self.vessel.req_latest_data()
        self._tmp_storage['cross_track_error'].append(abs(latest_data['navigation']['cross_track_error'])*100)

    def save_latest_episode(self, save_history=True):
        #print('Saving latest episode with save_history = ' + str(save_history))
        self.last_episode = {
            'path': self.path(np.linspace(0, self.path.length, 1000)) if self.path is not None else None,
            'path_taken': self.vessel.path_taken,
            'obstacles': self.obstacles
        }
        if save_history:
            self.history.append({
                'cross_track_error': np.array(self._tmp_storage['cross_track_error']).mean(),
                'reached_goal': int(self.reached_goal),
                'collision': int(self.collision),
                'reward': self.cumulative_reward,
                'timesteps': self.t_step,
                'duration': self.t_step*self.config["t_step_size"],
                'progress': self.progress,
                'pathlength': self.path.length
            })