
from setuptools import setup

setup(name='gym_auv',
      version='0.0.3',
      python_requires='>=3.7, <3.8',
      install_requires=[
          'beautifulsoup4==4.6.3',
          'box2d==2.3.2',
          'gast==0.2.2',
          'geopy==1.20.0',
          'gym==0.14.0',
          'keras==2.2.4',
          'matplotlib==3.0.2',
          'mpi4py>=3.0.2',
          'numpy==1.16.0',
          'numpy-stl==2.10.1',
          'opencv-python==4.1.0.25',
          'pybullet==2.5.5',
          'pygame==1.9.6',
          'pyglet==1.3.2',
          'python-utils==2.3.0',
          'pywavefront==1.3.1',
          'scikit-learn==0.20.1',
          'scipy==1.1.0',
          'stable-baselines==2.9.0', 
          'tensorflow==1.14.0',
          'wavefront-reader==0.2.2',
          'win10toast==0.9'
          ]  # And any other dependencies it needs
     )