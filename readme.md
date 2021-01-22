# gym-auv

Python simulation framework for Collision Avoidance for Unmanned Surface Vehicle using Deep Reinforcement Learning.

An explanation of the software structure can be found in Eivind Meyers repository [gym-auv](https://github.com/EivMeyer)

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


## Known bugs

* terrain.npy is missing from "resources/" because github does not support uploading large files. (reach out for a copy of this file)
* Lots of deprecation warnings because of TensorFlow 1
