# isaac_gym

## Create Isaac Gym Environment
Follow `isaac_gym_installation.pdf` to create the environment.

## Create Workspace
Clone the repo to anywhere you like in your workspace. Activate the environment and install `matplotlib` in the directory.
<br/>
```commandLine
$ conda activate rlgpu
$ (rlgpu) cd isaac_gym
$ (rlgpu) pip install matplotlib
```
Place the assets folder `resources` in the directory.

## Run Simulator
To run the simulator, just run:
```commandline
$ (rlgpu) python simulator.py
```

## Change the initial pose of robot
Go to `_create_robot()` in `env_creator.py`:
```python
robot_pose.p = gymapi.Vec3(0.0, 0.0, 0.1)
robot_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(0.0))  # in environment world frame
```
`robot_pose.p` is in `x, y, z`, to place the robot on different positions on the ground plane, just change the first two 
parameters. <br/>
`robot_pose.r` is the quaternion representation in `gymapi`. Specify the initial yaw angle of the robot in degrees.
Positive degree corresponds to rotating to the left.