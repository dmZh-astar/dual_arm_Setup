# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Mock dual-arm robot for testing planner_server.py without hardware.

Publishes ``/joint_states`` starting from the robot config's default joint position and
serves both arms' ``FollowJointTrajectory`` actions. Each trajectory is followed in time, so
the next goal starts where the last one ended. No physics, limits or collision checks.

    python scripts/mock_robot.py                    # terminal 1
    python scripts/planner_server.py                # terminal 2
    python scripts/send_goal.py --goal-poses scripts/goals.yml   # terminal 3
"""
import argparse
import threading
import time
from pathlib import Path
from typing import Dict, List

import rclpy
import yaml
from control_msgs.action import FollowJointTrajectory
from curobo._src.util_file import get_robot_configs_path
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState

from ros_trajectory_executor import ARM_CONTROLLERS, JOINT_STATE_TOPIC

_PUBLISH_RATE_HZ: float = 50.0


def load_default_joint_positions(robot_file: str) -> Dict[str, float]:
    """Read ``cspace.default_joint_position`` from a cuRobo robot config."""
    path = Path(robot_file)
    if not path.is_file():
        path = Path(get_robot_configs_path()) / robot_file
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Configs keep kinematics either at the top level or under robot_cfg.
    cspace = cfg.get("robot_cfg", cfg)["kinematics"]["cspace"]
    return dict(zip(cspace["joint_names"], cspace["default_joint_position"]))


class MockRobot(Node):
    """Publishes joint states and moves the joints along received trajectories."""

    def __init__(self, positions: Dict[str, float], time_scale: float):
        super().__init__("mock_robot")
        self._positions = dict(positions)
        self._lock = threading.Lock()
        self._time_scale = time_scale
        group = ReentrantCallbackGroup()
        self._pub = self.create_publisher(JointState, JOINT_STATE_TOPIC, 10)
        self.create_timer(1.0 / _PUBLISH_RATE_HZ, self._publish, callback_group=group)
        self._servers = [
            ActionServer(
                self,
                FollowJointTrajectory,
                f"/{name}/follow_joint_trajectory",
                execute_callback=self._make_execute(name, joints),
                callback_group=group,
            )
            for name, joints in ARM_CONTROLLERS.items()
        ]
        self.get_logger().info(f"Mock robot ready, publishing {JOINT_STATE_TOPIC}")

    def _publish(self) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        with self._lock:
            msg.name = list(self._positions)
            msg.position = list(self._positions.values())
        self._pub.publish(msg)

    def _make_execute(self, name: str, joints: List[str]):
        def execute(goal_handle) -> FollowJointTrajectory.Result:
            result = FollowJointTrajectory.Result()
            traj = goal_handle.request.trajectory
            if sorted(traj.joint_names) != sorted(joints):
                goal_handle.abort()
                result.error_code = FollowJointTrajectory.Result.INVALID_JOINTS
                result.error_string = f"expected joints {joints}, got {list(traj.joint_names)}"
                return result

            self.get_logger().info(f"{name}: following {len(traj.points)} points")
            start = time.monotonic()
            for point in traj.points:
                t = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
                delay = start + t / self._time_scale - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                with self._lock:
                    self._positions.update(zip(traj.joint_names, point.positions))

            goal_handle.succeed()
            result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
            self.get_logger().info(f"{name}: done")
            return result

        return execute


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Mock dual-arm robot for testing")
    parser.add_argument(
        "--robot",
        type=str,
        default="dual_arm.yml",
        help="Robot config file for the start pose (default: dual_arm.yml)",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="Play trajectories this many times faster than real time (default: 1.0)",
    )
    args = parser.parse_args()

    rclpy.init()
    node = MockRobot(load_default_joint_positions(args.robot), args.time_scale)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
