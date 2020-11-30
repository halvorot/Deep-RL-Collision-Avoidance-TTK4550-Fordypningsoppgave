# gym-auv

Python simulation framework for Collision Avoidance for Unmanned Surface Vehicle using Deep Reinforcement Learning


## Getting Started
Note: Requires Python 3.7

Note: Pybullet needs Microsoft Visual C++ 14.0. Install it with "Build Tools for Visual Studio".

Note: Stable-Baselines only supports Tensorflow 1.14, Tensorflow 2 support is planned. 

! Install Microsoft MPI (https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi) (msmpisetup.exe , not SDK)

Note: Run the following __first__.
```
conda install -c conda-forge shapely
conda install swig
conda install ffmpeg
```

Then run 

```
pip install -e ./gym-auv/
```

You can now execute the script by running 
```
python run.py <mode> <env>
``` 
The run script can be executed with the -h flag for a comprehensive overview of the available usage modes.

Examples:
```
python run.py play TestScenario1-v0
``` 
```
python run.py train MovingObstaclesNoRules-v0
``` 
```
python run.py enjoy MovingObstaclesNoRules-v0 --algo algorithm --agent path\to\agent.pkl
``` 


## Background

This Python package, which provides an easily expandable code framework for simulating autonomous surface vessels
in maritime environments, as well as training reinforcement learning-based AI agents to guide them, was developed as a part of my Master's thesis at the Norwegian University of Science and Technology.

Apart from the underlying simulation engine, which simulates the vessel dynamics according to well-researched manuevering theory,
as well as the functioning of a LiDAR-based sensor suite for distance measurements.
it also provides easy-to-use rendering in both 2D and 3D for debugging and showcasing purposes. Implemented as an extention of the OpenAI gym toolkit, it offers an easy-to-use interface for training state-of-the-art deep reinforcement learning algorithms for guiding the vessel.

The research paper [Taming an Autonomous Surface Vehicle for Path Following and Collision Avoidance Using Deep Reinforcement Learning (2020)](https://ieeexplore.ieee.org/document/9016254?fbclid=IwAR3obkbKJcbA2Jrn3nqKp7iUD_MAag01YSCm3liaIYJN7xN9enzdHUA0Ma8) gives a comprehensive overview of what the package is intended for.

>  In this article, we explore the feasibility of applying proximal policy optimization, a state-of-the-art deep reinforcement learning algorithm for continuous control tasks, on the dual-objective problem of controlling an underactuated autonomous surface vehicle to follow an a priori known path while avoiding collisions with non-moving obstacles along the way. The AI agent, which is equipped with multiple rangefinder sensors for obstacle detection, is trained and evaluated in a challenging, stochastically generated simulation environment based on the OpenAI gym Python toolkit. Notably, the agent is provided with real-time insight into its own reward function, allowing it to dynamically adapt its guidance strategy. Depending on its strategy, which ranges from radical path-adherence to radical obstacle avoidance, the trained agent achieves an episodic success rate close to 100%.



## Structure
The core component of **gym** is the environment abstraction Env, which represents the generalized RL environment . Notably, **gym** does not include a built-in Agent class of any kind. Instead, all the fundamental functionality required for an RL application, i.e. agent perception, reward calculation and action execution / environment updates are handled by the Env instance. Fundamentally, extensions of **gym**, including our **gym-auv** package, implement a subclass of `gym.Env` which overrides the core abstract methods: \_\_init\_\_, which defines the environment’s action and observation space; step, which simulates the environment for one timestep after an action has been performed and returns the observation vector and reward; reset, which resets the environment state to the initial state; and render, which renders the environment to the screen. In our case, this class is
named BaseEnvironment.

Furthermore, our framework uses three other classes, namely Vessel, Path, BaseObstacle and BaseRewarder. This provides a clear modular structure for the software and allows us to abstract away tedious function implementations. Also, it facilitates further extensions, such as adding a new vessel type with other dynamics, adding new obstacle shapes or introducing a new reward function to achieve different vessel behaviors. In the following, we will outline the details of these classes and how they are related.

### Path
The Path class represents an a priori available trajectory which is intended to be followed by a Vessel instance. It provides not only a lookup method mapping from a
specified arc-length value to the corresponding coordinate point, but also helper methods
that facilitate a vessel’s navigation with the respect to the path. In the default behavior,
a smooth trajectory parameterized by arc length is generated using 1D Piecewise Cubic
Hermite Interpolator (PCHIP) provided by SciPy (67) based on the waypoints argument required by the constructor method. Optionally, by calling the constructor with the
keyword smooth=False, the user can also create a path made of linear line segments
connecting the specified way-points.

### BaseObstacle
The BaseObstacle class is an abstract class that represents physical obstacles that a Vessel instance can collide with and should avoid. Due to the vast variety of obstacles the user might be interested in using in a scenario, both in terms of shape and dynamic properties, it is designed as an abstract class which is intended to be implemented by its sub-classes. As will be discussed later in this chapter, the gym-auv package relies upon the Python package Shapely for the geometric operations required to simulate a rangefinder sensor suite. Thus, the BaseObstacle class’ public boundary attribute, which has the type of a shapely.geometry.Polygon, is a critical feature of the class as it is required for simulating the rangefinder sensors’ detection of the obstacle. Also, the BaseObstacle class includes an update wrapper method for updating the obstacle’s position given its dynamic properties - for instance given its speed and heading if the obstacle represents another vessel. The specific update behavior must be implemented in extensions of BaseObstacle, and can be left blank in the case of static obstacles.

### Vessel
The Vessel class represents a physical vessel placed in an environment. This should not be confused with the agent as thought of in an RL-context - an autonomous entity directing its actions towards achieving its goals within its environment. In our gym-auv framework, the Vessel class is simply responsible for updating the vessel state according to the ship dynamics as well as simulating the sensor suite attached to the vessel. This functionality logically belongs to the environment module, but is implemented as a separate module to facilitate a possible multi-agent use-case with several vessels interacting within the same environment.

### BaseRewarder
The BaseRewarder class is responsible for calculating the reward received by an agent at each time step. It is designed as an abstract class which is intended to be implemented by its sub-classes. As the reward, in the general case, depends on a vessel’s adherence to its desired path as well as its distance from obstacles in its proximity, a BaseRewarder instance gets a Vessel instance assigned to it in the constructor which it accesses upon calculating the reward. As was the case for the Vessel class, the motivation for detaching this functionality from the BaseEnvironment is to facilitate multi-agent extensions
with possible nonuniform agent objectives, necessitating a one-to-many relationship between environment and rewarder.

### BaseEnvironment
Extending the gym.Env base environment class, BaseEnvironment is the access point for using third-party RL algorithms to train agents in our environment. It implements the core abstract gym.Env methods __init__, reset, step and render. Notably, it also specifies its own abstract method to be implemented by specific scenario implementations, namely that of _generate, which, as the name suggests, (randomly) creates a new obstacle environment and is called each time the environment resets.

## Author
* **Halvor Ødegård Teigen** - [halvorot](https://github.com/halvorot)
* **Eivind Meyer** - [EivMeyer](https://github.com/EivMeyer)

## Screenshots

![3D Rendering](https://i.imgur.com/KD0TqZW.png)

![2D Rendering](https://i.imgur.com/dBQOWYT.png)


## Known bugs

* TestScenarios and DebugScenario have constant reward while training
* TestScenarios and DebugScenario gets -5000 constant reward after crash in play mode
* Lots of deprecation warnings because of TensorFlow 1