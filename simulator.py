import numpy as np
import os
import shutil
from isaacgym import gymapi, gymutil, gymtorch
import math
import torch
import matplotlib.pyplot as plt

from PIL import Image

from assets_loader import load_terrasentia, load_plants, load_ground
from env_creator import EnvCreator
from mpc import MPC
from mpc_handle import MPCHandle


class Simulator:
    def __init__(self):
        self.mpc = MPC()
        self.mpc_handle = MPCHandle(self.mpc)

    @staticmethod
    def euler_from_quaternion(x, y, z, w):
        """Convert a quaternion into euler angles (roll, pitch, yaw)

        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z  # in radians

    def run_simulator(self):
        """The main program to run simulator
        """
        gym = gymapi.acquire_gym()  # Initialize gym
        args = gymutil.parse_arguments(description="terrasentia_env")  # Parse arguments

        # Create Simulation
        sim_params = gymapi.SimParams()
        sim_params.dt = dt = 1.0 / 60.0
        sim_params.up_axis = gymapi.UP_AXIS_Z  # Specify z-up coordinate system
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        sim_params.use_gpu_pipeline = True  # Use GPU pipeline
        sim_params.physx.use_gpu = True
        if args.physics_engine == gymapi.SIM_FLEX:
            pass
        elif args.physics_engine == gymapi.SIM_PHYSX:
            sim_params.physx.solver_type = 1
            sim_params.physx.num_position_iterations = 6
            sim_params.physx.num_velocity_iterations = 0
            sim_params.physx.num_threads = args.num_threads
        if args.use_gpu_pipeline:
            print("WARNING: Forcing CPU pipeline.")
        sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
        if sim is None:
            print("*** Failed to create sim")
            quit()

        # Add Ground Plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up!
        gym.add_ground(sim, plane_params)

        # Create Global Camera Viewer
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            print("*** Failed to create viewer")
            quit()
        cam_pos = gymapi.Vec3(-4.0, -1.0, 4.0)
        cam_target = gymapi.Vec3(0.0, 1.0, 2.0)
        gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

        # Load Assets
        print("Loading assets...")
        robot_asset = load_terrasentia(gym, sim)
        plants_assets_dict = load_plants(gym, sim)
        ground_asset = load_ground(gym, sim)

        # DOF Initialization
        def clamp(x, min_value, max_value):
            return max(min(x, max_value), min_value)

        dof_names = gym.get_asset_dof_names(robot_asset)  # Get array of DOF names
        dof_props = gym.get_asset_dof_properties(robot_asset)  # Get array of DOF properties
        num_dofs = gym.get_asset_dof_count(robot_asset)  # Create an array of DOF states
        dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
        dof_types = [gym.get_asset_dof_type(robot_asset, i) for i in range(num_dofs)]  # Get list of DOF types
        dof_positions = dof_states['pos']  # Get the position slice of the DOF state array
        # Get the limit-related slices of the DOF properties array
        stiffnesses = dof_props['stiffness']
        dampings = dof_props['damping']
        armatures = dof_props['armature']
        has_limits = dof_props['hasLimits']
        lower_limits = dof_props['lower']
        upper_limits = dof_props['upper']

        # Initialize default positions, limits, and speeds (make sure they are in reasonable ranges)
        defaults = np.zeros(num_dofs)
        speeds = np.zeros(num_dofs)
        for i in range(num_dofs):
            if has_limits[i]:
                lower_limits[i] = clamp(lower_limits[i], -math.pi, math.pi)
                upper_limits[i] = clamp(upper_limits[i], -math.pi, math.pi)
                # Make sure our default position is in range
                if lower_limits[i] > 0.0:
                    defaults[i] = lower_limits[i]
                elif upper_limits[i] < 0.0:
                    defaults[i] = upper_limits[i]
            else:
                # Set reasonable animation limits for unlimited joints
                # Unlimited revolute joint
                lower_limits[i] = -math.pi
                upper_limits[i] = math.pi
            dof_positions[i] = defaults[i]  # Set DOF position to default

        for i in range(num_dofs):
            print("DOF %d" % i)
            print("  Name:     '%s'" % dof_names[i])
            print("  Type:     %s" % gym.get_dof_type_string(dof_types[i]))
            print("  Stiffness:  %r" % stiffnesses[i])
            print("  Damping:  %r" % dampings[i])
            print("  Armature:  %r" % armatures[i])
            print("  Limited?  %r" % has_limits[i])
            if has_limits[i]:
                print("    Lower   %f" % lower_limits[i])
                print("    Upper   %f" % upper_limits[i])

        # Create Environment(s)
        num_envs = 1
        num_per_row = 1
        spacing = 20.0
        env_creator = EnvCreator(gym, sim, dof_states, robot_asset, plants_assets_dict, ground_asset,
                                 num_envs, num_per_row, spacing)
        envs, actor_handles, camera_handles = env_creator.create_env()

        # Prepare simulation for GPU (tensor API)
        gym.prepare_sim(sim)
        _root_tensor = gym.acquire_actor_root_state_tensor(sim)  # Acquire root state tensor descriptor
        root_tensor = gymtorch.wrap_tensor(_root_tensor)  # Wrap it in a PyTorch Tensor and create convenient views
        root_positions = root_tensor[:, 0:3]
        root_orientations = root_tensor[:, 3:7]
        root_linvels = root_tensor[:, 7:10]
        root_angvels = root_tensor[:, 10:13]

        # Create Path to Save Images
        if os.path.exists("graphics_images"):
            shutil.rmtree("graphics_images")
        os.mkdir("graphics_images")
        for i in range(num_envs):
            if not os.path.exists(f"graphics_images/env{i}"):
                os.mkdir(f"graphics_images/env{i}")

        # Run Simulation
        frame_count = 0
        
        # Plotting usage
        mpc_linear = []
        gt_linear = []
        gt_angular = []
        mpc_angular_z = []
        x_body = []
        y_body = []
        
        while not gym.query_viewer_has_closed(viewer):
            # Step the physics
            gym.simulate(sim)
            gym.refresh_actor_root_state_tensor(sim)  # Refresh the state tensors
            gym.fetch_results(sim, True)

            # Retrieve robot current position and orientation
            cur_robot_pos = root_positions[0]
            cur_robot_orient = root_orientations[0]
            x_pos = cur_robot_pos[0].item()
            y_pos = cur_robot_pos[1].item()
            x_quat = cur_robot_orient[0].item()
            y_quat = cur_robot_orient[1].item()
            z_quat = cur_robot_orient[2].item()
            w_quat = cur_robot_orient[3].item()
            row, pitch, yaw = self.euler_from_quaternion(x_quat, y_quat, z_quat, w_quat)

            x_body_init, y_body_init = self.mpc_handle.prepare_ref_path(x_pos, y_pos, yaw)  # Prepare reference path
            u, mpc_output, ss_error = self.mpc.solve_mpc(self.mpc_handle.mpc_reference)  # MPC output

            # Convert body velocity to wheel velocity
            linear_x_vel = u[0, 0]
            angular_z_vel = - u[1, 0]  # TODO: Why negative sign here?
            right_angular, left_angular = self.mpc_handle.body_to_wheel_vel(linear_x_vel, - angular_z_vel)

            # print("%f, %f, %f" % (row, pitch, yaw))
            # print('Position: ', cur_robot_pos)s
            # print(self.mpc_handle.mpc_reference)
            # print('linear_x_vel', linear_x_vel)
            # print('angular_z_vel', angular_z_vel)
            # print('right_angular', right_angular)
            # print('left_angular', left_angular)
            # print('ground_truth_lin', root_linvels[0])
            
            # Plotting usage
            mpc_linear.append(float(linear_x_vel))
            mpc_angular_z.append(-float(angular_z_vel))
            gt_linear.append(np.sqrt((float(root_linvels[0][0].item())) ** 2 + (float(root_linvels[0][1].item())) ** 2))
            gt_angular.append(float(root_angvels[0][2].item()))  # Angular vel for z-axis
            x_body.append(float(x_body_init))
            y_body.append(float(y_body_init))

            # Drive DOF with velocity from MPC (tensor control)
            num_dofs = gym.get_sim_dof_count(sim)  # Total number of DOFs for all environments
            vel_targets = torch.zeros(num_dofs, dtype=torch.float32, device='cuda:0')
            for i in range(0, num_dofs, 8):
                vel_targets[i + 1] = float(left_angular)  # unit: radians/second
                vel_targets[i + 3] = float(right_angular)
                vel_targets[i + 5] = float(left_angular)
                vel_targets[i + 7] = float(right_angular)
            gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(vel_targets))

            # Update the viewer
            gym.step_graphics(sim)

            # Render cameras
            gym.render_all_camera_sensors(sim)
            # Save images
            for i in range(num_envs):
                camera_handle = camera_handles[i][0]
                camera_intrinsic = camera_handles[i][1]  # (3, 3) ndarray
                img_width = camera_handles[i][2]
                img_height = camera_handles[i][3]
                if frame_count % 100 == 0:
                    # Retrieve image data directly
                    # TODO: Adjust camera principal point according to camera intrinsic
                    rgb_image = gym.get_camera_image(sim, envs[i], camera_handle, gymapi.IMAGE_COLOR)
                    rgb_image = np.reshape(rgb_image, (img_height, img_width, 4))
                    image = Image.fromarray(rgb_image)
                    image.save(f"graphics_images/env{i}/rgb_env{i}_frame{frame_count}.png")
            frame_count += 1

            gym.draw_viewer(viewer, sim, True)
            # Wait for dt to elapse in real time.
            # This synchronizes the physics simulation with the rendering rate.
            gym.sync_frame_time(sim)
        
        # Plotting usage
        time_steps = [i for i in range(frame_count)]
        # Plot output linear velocity and ground truth linear velocity
        plt.figure()
        plt.plot(time_steps, mpc_linear, color='g', label="MPC Linear Velocity")
        plt.plot(time_steps, gt_linear, color='r', label="Ground-truth Linear Velocity")
        plt.xlabel('Time Steps')
        plt.ylabel('Linear Velocity')
        plt.title('MPC v.s. Ground-truth', fontsize=20)
        plt.grid()
        plt.legend()
        plt.savefig(f'lin_vel.jpg')

        # Plot output angular velocity and ground truth angular velocity
        plt.figure()
        plt.plot(time_steps, mpc_angular_z, color='g', label="MPC Angular Velocity")
        plt.plot(time_steps, gt_angular, color='r', label="Ground-truth Angular Velocity")
        plt.xlabel('Time Steps')
        plt.ylabel('Angular Velocity')
        plt.title('MPC v.s. Ground-truth', fontsize=20)
        plt.grid()
        plt.legend()
        plt.savefig(f'ang_vel.jpg')

        # Plot the first reference point for every time step
        plt.figure()
        plt.plot(time_steps, x_body, color='g', label="x")
        plt.plot(time_steps, y_body, color='r', label="y")
        plt.xlabel('Time Steps')
        plt.ylabel('Body Positions')
        plt.title('Body x-y Positions 30Deg', fontsize=20)
        plt.grid()
        plt.legend()
        plt.savefig(f'positions.jpg')

        # Shut down simulator
        gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)


if __name__ == "__main__":
    simulator = Simulator()
    simulator.run_simulator()


