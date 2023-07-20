import numpy as np
import random
from isaacgym import gymapi


class EnvCreator:
    """This class is used to handle creating simulator environment by creating actors:
    - Create and place colored ground plane
    - Create and place terrasentia
    - Create and place plants from randomized parameters
    """
    def __init__(self, gym, sim, dof_states, robot_asset, plants_assets_dict, ground_asset,
                 num_envs, num_per_row, spacing):
        self.gym = gym
        self.sim = sim
        self.dof_states = dof_states
        self.robot_asset = robot_asset
        self.plants_assets_dict = plants_assets_dict
        self.ground_asset = ground_asset

        # Environments, actors, and camera handles
        self.envs = []  # [env_0, env_1, ...]
        self.actor_handles = []  # [[ground_handle, robot_handle, plant_handle_1, ...], ...]
        self.camera_handles = []  # [[cam_1_handle, cam_1_intrinsic, img_width, img_height], ...]

        # Static parameters
        self.num_envs = num_envs
        self.num_per_row = num_per_row
        self.spacing = spacing
        self.env_lower = gymapi.Vec3(-self.spacing, 0.0, -self.spacing)
        self.env_upper = gymapi.Vec3(self.spacing, self.spacing, self.spacing)

        # Constants for randomizing other parameters
        # TODO: Currently only randomizing plant type and orientation across envs
        self.plants_names = list(self.plants_assets_dict.keys())
        self.min_yaw_angle = -180
        self.max_yaw_angle = 180
        self.min_row_length = 2.0
        self.max_row_length = 5.0
        self.min_plants_dist = 0.2
        self.max_plants_dist = 0.4

        # Camera related
        # TODO: Currently not randomizing any camera parameters
        # TODO: Confirm all camera parameters range
        self.min_cam_height = -0.05
        self.max_cam_height = 0.05  # Relative to the camera_node rigid body
        self.min_horizontal_fov = 90  # Degrees
        self.max_horizontal_fov = 150
        self.min_img_width = 512
        self.max_img_width = 1024
        self.min_img_height = 256
        self.max_img_height = 512
        self.min_x_offset = 256
        self.max_x_offset = 512
        self.min_y_offset = 128
        self.max_y_offset = 256

    @staticmethod
    def _generate_random_green():
        """Generate random green RGB colors for plants.

        :return red_intensity: The R value of RGB code
        :type red_intensity: int
        :return red_intensity: The G value of RGB code
        :type red_intensity: int
        :return red_intensity: The B value of RGB code
        :type red_intensity: int
        """
        red_intensity = random.randint(0, 195)
        green_intensity = 255
        blue_intensity = random.randint(0, 128)
        return red_intensity, green_intensity, blue_intensity

    @staticmethod
    def _generate_random_brown():
        """Generate random brown RGB colors for the ground surface.

        :return red_intensity: The R value of RGB code
        :type red_intensity: int
        :return red_intensity: The G value of RGB code
        :type red_intensity: int
        :return red_intensity: The B value of RGB code
        :type red_intensity: int
        """
        brown_rgb = {
            'burlywood':  (222, 184, 135),
            'tan': (210, 180, 140),
            'sandybrown':  (244, 164, 96),
            'peru':  (205, 133, 63),
            'saddlebrown':  (139, 69, 19),
            'sienna': (160, 82, 45),
        }
        brown_names = list(brown_rgb.keys())
        brown_name = random.choice(brown_names)
        red_intensity, green_intensity, blue_intensity = brown_rgb[brown_name]
        return red_intensity, green_intensity, blue_intensity

    @staticmethod
    def _perform_bernoulli_trial(gap_prob):
        """Simulate a simple bernoulli trial given the gap probability at every step when placing plants.

        Return True if there will be a gap at current step;
        Return False if there will not be a gap at current step.
        """
        sample = random.uniform(0, 1)
        if sample < gap_prob:
            return True
        return False

    @staticmethod
    def _create_camera_intrinsic(img_width, img_height, horizontal_fov, x_offset, y_offset):
        """Construct the camera intrinsic for the camera sensor in every environment.

        :return camera_intrinsic: The camera intrinsic matrix (3, 3)
        :type camera_intrinsic: np.ndarray
        """
        vertical_fov = (img_height / img_width * horizontal_fov) * np.pi / 180
        horizontal_fov *= np.pi / 180

        f_x = (img_width / 2.0) / np.tan(horizontal_fov / 2.0)  # Focal length
        f_y = (img_height / 2.0) / np.tan(vertical_fov / 2.0)

        camera_intrinsic = np.array([[f_x, 0.0, x_offset],
                                     [0.0, f_y, y_offset],
                                     [0.0, 0.0, 1.0]])
        return camera_intrinsic

    def _create_ground(self, env, env_idx, env_actor_handles):
        """Place the ground asset in the current environment by creating the corresponding actor.

        :param env: An environment in the simulator
        :type env: environment class
        :param env_idx: The index of the environment
        :type env_idx: int
        :param env_actor_handles: A list containing all actor handlers in current environment
        :type env_actor_handles: list
        """
        # TODO: Fix ground color for now
        ground_rgb = (244, 164, 96)

        ground_pose = gymapi.Transform()
        ground_pose.p = gymapi.Vec3(10.0, 0.0, 0.0)
        ground_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1),
                                                    np.radians(-90.0))  # In environment world frame
        ground_handle = self.gym.create_actor(env, self.ground_asset, ground_pose, "ground_plane", env_idx, 1)
        ground_r, ground_g, ground_b = ground_rgb
        self.gym.set_rigid_body_color(env, ground_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                      gymapi.Vec3(ground_r / 255, ground_g / 255, ground_b / 255))
        env_actor_handles.append(ground_handle)

    def _create_robot(self, env, env_idx, env_actor_handles):
        """Place the Terrasentia asset in the current environment by creating the corresponding actor.

        :param env: An environment in the simulator
        :type env: environment class
        :param env_idx: The index of the environment
        :type env_idx: int
        :param env_actor_handles: A list containing all actor handlers in current environment
        :type env_actor_handles: list

        :return robot_handle: The robot actor handler
        :type robot_handle: actor class
        """
        robot_pose = gymapi.Transform()
        robot_pose.p = gymapi.Vec3(0.0, 0.0, 0.1)
        robot_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.radians(0.0))  # in environment world frame
        robot_handle = self.gym.create_actor(env, self.robot_asset, robot_pose, "terrasentia", env_idx, 2)
        env_actor_handles.append(robot_handle)
        return robot_handle

    def _create_plants(self, env, env_idx, env_actor_handles):
        """Place two rows of plants (the same type) in the current environment by creating the corresponding actor;
        The robot is between the two rows of plants.

        :param env: An environment in the simulator
        :type env: environment class
        :param env_idx: The index of the environment
        :type env_idx: int
        :param env_actor_handles: A list containing all actor handlers in current environment
        :type env_actor_handles: list
        """
        # TODO: Fix some of the plant parameters for now
        plant_name = random.choice(self.plants_names)
        plant_asset = self.plants_assets_dict[plant_name]  # Choose plant asset
        row_length = 10  # Length of a row (m)
        plants_dist = 0.2  # Distance between individual plants (m)
        gap_prob = 0.02
        plant_rgb = self._generate_random_green()  # Randomize plant colors (green range)

        # Place plants on the left and right of terrasentia in two separate rows
        cur_row_length = 0.0
        step_count = 0
        while cur_row_length <= row_length:
            # Randomize plants orientation for each step
            yaw_angle_list = []
            for i in range(4):
                yaw_angle_list.append(random.randint(-180, 180))

            if not self._perform_bernoulli_trial(gap_prob):
                # Left-hand first row
                plant_pose = gymapi.Transform()
                plant_pose.p = gymapi.Vec3(step_count * plants_dist, 0.3, 0.0)
                plant_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1),
                                                           np.radians(yaw_angle_list[0]))  # In environment world frame
                plant_handle = self.gym.create_actor(env, plant_asset, plant_pose,
                                                     f"{plant_name}_{step_count}_left", env_idx, 3)
                plant_r, plant_g, plant_b = plant_rgb
                self.gym.set_rigid_body_color(env, plant_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                              gymapi.Vec3(plant_r/255, plant_g/255, plant_b/255))
                env_actor_handles.append(plant_handle)

            if not self._perform_bernoulli_trial(0.02):
                # Right-hand first row
                plant_pose = gymapi.Transform()
                plant_pose.p = gymapi.Vec3(step_count * plants_dist, -0.3, 0.0)
                plant_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1),
                                                           np.radians(yaw_angle_list[2]))  # In environment world frame
                plant_handle = self.gym.create_actor(env, plant_asset, plant_pose,
                                                     f"{plant_name}_{step_count}_right", env_idx, 3)
                plant_r, plant_g, plant_b = plant_rgb
                self.gym.set_rigid_body_color(env, plant_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                              gymapi.Vec3(plant_r / 255, plant_g / 255, plant_b / 255))
                env_actor_handles.append(plant_handle)

            # Update step
            step_count += 1
            cur_row_length += plants_dist

    def _create_cameras(self, env_idx):
        """Create a camera sensor in the current environment; The camera is attached to the front of the robot.

        :param env_idx: The index of the environment
        :type env_idx: int
        """
        # TODO: Fix camera parameters for now
        camera_height = 0.0
        img_width = 500
        img_height = 500
        camera_intrinsic = np.ones((3, 3))

        camera_props = gymapi.CameraProperties()
        camera_props.width = img_width
        camera_props.height = img_height

        # Attach camera to terrasentia
        camera_handle = self.gym.create_camera_sensor(self.envs[env_idx], camera_props)
        camera_offset = gymapi.Vec3(0.1, 0.0, camera_height)  # In camera node body frame
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0),
                                                      np.radians(0.0))  # In environment world frame
        robot_handle = self.actor_handles[env_idx][0]
        body_handle = self.gym.get_actor_rigid_body_handle(self.envs[env_idx], robot_handle,
                                                           10)  # Attach to 10th rigid body (camera node)

        self.gym.attach_camera_to_body(camera_handle, self.envs[env_idx], body_handle,
                                       gymapi.Transform(camera_offset, camera_rotation),
                                       gymapi.FOLLOW_TRANSFORM)
        self.camera_handles.append([camera_handle, camera_intrinsic, img_width, img_height])

    def create_env(self):
        """Create multiple environments in the simulator, specified by the self.num_env parameter; In each
        environment, create ground, Terrasentia, and plants actors. In addition, create a camera handler in each
        environment. The function also specifies the control mode of the robot.

        :return self.envs: A list storing all the environments objects
        :type self.envs: list
        :return self.actor_handles: A nested list storing all actors in the simulator
        :type self.actor_handles: list
        :return self.camera_handles: A list storing camera handles for all environments
        :type self.camera_handles: list
        """
        print("Creating %d environments..." % self.num_envs)
        for i in range(0, self.num_envs):
            env = self.gym.create_env(self.sim, self.env_lower, self.env_upper, self.num_per_row)
            env_actor_handles = []  # Multiple actors in one environment

            robot_handle = self._create_robot(env, i, env_actor_handles)  # Create terrasentia actor
            self.gym.set_actor_dof_states(env, robot_handle, self.dof_states,
                                          gymapi.STATE_ALL)  # Set default DOF positions for terrasentia
            self._create_plants(env, i, env_actor_handles)  # Create plants actors
            self._create_ground(env, i, env_actor_handles)  # Create ground
            self.envs.append(env)
            self.actor_handles.append(env_actor_handles)

            # Enable Velocity Control Mode
            props = self.gym.get_actor_dof_properties(env, robot_handle)
            props["driveMode"].fill(gymapi.DOF_MODE_VEL)
            props["stiffness"].fill(0.0)
            props["damping"].fill(600.0)
            self.gym.set_actor_dof_properties(env, robot_handle, props)

            self._create_cameras(i)  # Create camera attached on terrasentia

        return self.envs, self.actor_handles, self.camera_handles
