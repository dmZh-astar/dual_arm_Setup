# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""ROS 2 node that reads the robot state and executes trajectories on both arm controllers."""
import time
from typing import Dict, List, Optional

import numpy as np
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState as RosJointState
from trajectory_msgs.msg import JointTrajectoryPoint

JOINT_STATE_TOPIC: str = "/joint_states"
_JOINT_STATE_TIMEOUT_SEC: float = 5.0
_ACTION_SERVER_TIMEOUT_SEC: float = 10.0
# Extra time allowed on top of the trajectory duration before giving up on a result.
_RESULT_MARGIN_SEC: float = 5.0

# Arm controllers from ros2_controllers.yaml, with joints in the order the controller expects.
ARM_CONTROLLERS: Dict[str, List[str]] = {
    "left_arm_controller": [
        "left_shoulder_pan_joint",
        "left_shoulder_lift_joint",
        "left_elbow_joint",
        "left_wrist_1_joint",
        "left_wrist_2_joint",
        "left_wrist_3_joint",
    ],
    "right_arm_controller": [
        "right_shoulder_pan_joint",
        "right_shoulder_lift_joint",
        "right_elbow_joint",
        "right_wrist_1_joint",
        "right_wrist_2_joint",
        "right_wrist_3_joint",
    ],
}


class CuRoboPlanner(Node):
    """Reads the robot state and sends cuRobo trajectories to the arm controllers."""

    def __init__(self):
        super().__init__("curobo_planner")
        self._latest_positions: Dict[str, float] = {}
        self.create_subscription(RosJointState, JOINT_STATE_TOPIC, self._on_joint_state, 10)
        self._traj_clients = {
            name: ActionClient(self, FollowJointTrajectory, f"/{name}/follow_joint_trajectory")
            for name in ARM_CONTROLLERS
        }

    def _on_joint_state(self, msg: RosJointState) -> None:
        self._latest_positions.update(zip(msg.name, msg.position))

    def wait_for_joint_positions(
        self, joint_names: List[str], timeout_sec: float = _JOINT_STATE_TIMEOUT_SEC
    ) -> Optional[Dict[str, float]]:
        """Wait until every joint in *joint_names* has been reported, or time out.

        The node must be spinning in another thread, as in every blocking method here.
        """
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if all(n in self._latest_positions for n in joint_names):
                return dict(self._latest_positions)
            time.sleep(0.05)
        missing = [n for n in joint_names if n not in self._latest_positions]
        self.get_logger().error(f"No {JOINT_STATE_TOPIC} data for joints: {missing}")
        return None

    def _wait_for(self, future, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        return future.done()

    def execute(
        self, positions: np.ndarray, velocities: np.ndarray, dt: float, joint_names: List[str]
    ) -> bool:
        """Send the trajectory to both arm controllers at once and wait for both results.

        Args:
            positions: Joint positions, shape ``(n_waypoints, dof)``. Waypoint 0 is the
                start state and is skipped.
            velocities: Joint velocities, shape ``(n_waypoints, dof)``.
            dt: Time between waypoints in seconds.
            joint_names: Joint name for each column of *positions*.

        Returns:
            True if both controllers report success.
        """
        for name, client in self._traj_clients.items():
            if not client.wait_for_server(timeout_sec=_ACTION_SERVER_TIMEOUT_SEC):
                self.get_logger().error(f"Action server for {name} not available")
                return False

        goals = {}
        for name, arm_joints in ARM_CONTROLLERS.items():
            idx = [joint_names.index(j) for j in arm_joints]
            goal = FollowJointTrajectory.Goal()
            goal.trajectory.joint_names = arm_joints
            for k in range(1, positions.shape[0]):
                point = JointTrajectoryPoint()
                point.positions = positions[k, idx].tolist()
                point.velocities = velocities[k, idx].tolist()
                point.time_from_start = Duration(nanoseconds=int(k * dt * 1e9)).to_msg()
                goal.trajectory.points.append(point)
            goals[name] = goal

        send_futures = {
            name: self._traj_clients[name].send_goal_async(goal) for name, goal in goals.items()
        }
        handles = {}
        for name, future in send_futures.items():
            if self._wait_for(future, _ACTION_SERVER_TIMEOUT_SEC) and future.result().accepted:
                handles[name] = future.result()
            else:
                self.get_logger().error(f"{name} rejected the trajectory")
        if len(handles) != len(goals):
            for handle in handles.values():
                handle.cancel_goal_async()
            return False

        self.get_logger().info("Both arms executing...")
        timeout = positions.shape[0] * dt + _RESULT_MARGIN_SEC
        success = True
        for name, handle in handles.items():
            result_future = handle.get_result_async()
            if not self._wait_for(result_future, timeout):
                self.get_logger().error(f"{name} did not finish in {timeout:.1f}s")
                success = False
                continue
            result = result_future.result().result
            if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
                self.get_logger().error(
                    f"{name} failed: error_code={result.error_code} {result.error_string}"
                )
                success = False
        return success
