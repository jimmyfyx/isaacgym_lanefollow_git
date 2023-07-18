import numpy as np


class MPCHandle:
    def __init__(self, mpc_controller):
        self.mpc = mpc_controller

        # Initialize reference input
        self.mpc_reference = {'x': [], 'y': [], 'theta': [], 'speed': []}
        self.ref_path_length = 10
        self.dt = 0.2
        self.ref_speed = 1.0  # m/s

        self.base_length = self.mpc.p['Lbase']
        self.wheel_radius = 0.09  # meters

    def prepare_ref_path(self, x_pos, y_pos, yaw):
        """Generate reference path for MPC as a list of 2D waypoints (x, y, theta)

        :param x_pos: Current robot x position in world frame
        :type x_pos: float
        :param y_pos: Current robot y position in world frame
        :type y_pos: float
        :param yaw: Current robot yaw angle relative to world frame
        :type yaw: float
        """
        # Clear previous reference path
        self.mpc_reference['x'].clear()
        self.mpc_reference['y'].clear()
        self.mpc_reference['theta'].clear()

        x_1, y_1 = (x_pos, y_pos)  # Robot body frame origin in world frame
        # Find another point on robot's body y-axis
        dx = - np.sin(yaw) * 0.5
        dy = np.cos(yaw) * 0.5
        x_2, y_2 = (x_pos + dx, y_pos + dy)

        # Solve the equation of the line connecting point1 and point2 in world frame
        # TODO: Edge case when body y-axis is parallel to the center of lane (robot never in this pose?)
        slope = (y_2 - y_1) / (x_2 - x_1)
        intersect = y_2 - slope * x_2

        # Solve for the point body y-axis intersects with world x-axis, in world frame
        x_ref_world = - intersect / slope  # The first point of reference path in world frame
        y_ref_world = 0.0

        # Solve transformation matrix from body to world frame
        rot_matrix = np.array([[np.cos(yaw), - np.sin(yaw)],
                               [np.sin(yaw), np.cos(yaw)]])
        rot_matrix = np.linalg.inv(rot_matrix)  # Represent world frame in body frame
        trans_vec = np.array([-x_pos, -y_pos, 1])
        trans_matrix = np.zeros((3, 3))
        trans_matrix[0:2, 0:2] = rot_matrix
        trans_matrix[:, 2] = trans_vec
        body_homo_coordinate = np.matmul(trans_matrix, np.array([[x_ref_world], [y_ref_world], [1]]))
        x_body = body_homo_coordinate[0][0]  # The first point of reference path in body frame
        y_body = body_homo_coordinate[1][0]

        print('x_world', x_ref_world)
        print('y_world', y_ref_world)
        print('x_body', x_body)
        print('y_body', y_body)

        # First reference waypoint
        self.mpc_reference['x'].append(x_body)  # TODO: Try not setting body x to 0
        self.mpc_reference['y'].append(y_body)
        self.mpc_reference['theta'].append(-yaw)  # TODO: What does theta mean here?
        self.mpc_reference['speed'] = np.ones(self.ref_path_length + 1) * self.ref_speed  # Reference speed

        # Prepare the other reference points (follows the center lane)
        for i in range(1, self.ref_path_length + 1):
            x_world = x_ref_world + i * (self.ref_speed * self.dt)
            y_world = 0.0
            body_homo_coordinate = np.matmul(trans_matrix, np.array([[x_world], [y_world], [1]]))
            x_body = body_homo_coordinate[0][0]
            y_body = body_homo_coordinate[1][0]
            self.mpc_reference['x'].append(x_body)
            self.mpc_reference['y'].append(y_body)
            self.mpc_reference['theta'].append(-yaw)  # TODO: What does theta mean here?

    def body_to_wheel_vel(self, body_x_lin, body_z_ang):
        """Use differential drive model to convert body velocities to wheel velocities

        :param body_x_lin: Robot body linear velocity
        :type body_x_lin: float
        :param body_z_ang: Robot angular velocity
        :type body_z_ang: float

        :return right_angular: Angular velocity for right wheel
        :type right_angular: float
        :return left_angular: Angular velocity for left wheel
        :type left_angular: float
        """
        right_linear = (body_z_ang * self.base_length + 2 * body_x_lin) / 2
        left_linear = (2 * body_x_lin - body_z_ang * self.base_length) / 2
        right_angular = right_linear / self.wheel_radius
        left_angular = left_linear / self.wheel_radius
        return right_angular, left_angular
