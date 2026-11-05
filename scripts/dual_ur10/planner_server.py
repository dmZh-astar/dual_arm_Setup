# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Long-running cuRobo planning server for the dual-arm robot.

Warms up cuRobo once, then waits on the goal topic. Each goal message triggers one
plan-and-execute: read ``/joint_states``, plan through the goals, and send one
``FollowJointTrajectory`` goal per arm controller.

Goal topic (``geometry_msgs/PoseArray``): poses for every tool frame of every goal, goal by
goal in tool-frame order, i.e. ``[goal1 left_tcp, goal1 right_tcp, goal2 left_tcp, ...]``,
in the cuRobo world frame. Keep an arm still by repeating its pose. Goals that arrive while a
motion is running are rejected. Progress is published on the status topic
(``std_msgs/String``): ``ready``, ``planning``, ``executing``, ``succeeded``, ``failed: ...``,
``rejected: ...``. Statuses about a goal start with that goal's ``header.stamp`` as
``[sec.nanosec]`` so senders can tell their goal apart from others.

    python scripts/planner_server.py --speed 0.1
    python scripts/send_goal.py                      # publish the built-in goals
"""
import argparse
import queue
import threading
from typing import Dict, List, Optional

import rclpy
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import JointState
from geometry_msgs.msg import PoseArray
from std_msgs.msg import String

from dual_arm_planning import DEFAULT_SPEED, goals_to_goal_tool_pose, multi_phase_planning
from ros_trajectory_executor import ARM_CONTROLLERS, JOINT_STATE_TOPIC, CuRoboPlanner

GOAL_TOPIC: str = "/curobo_planner/goal_poses"
STATUS_TOPIC: str = "/curobo_planner/status"
# cuRobo plans in the robot config's base_link frame ("world" for dual_arm.yml).
_PLANNING_FRAME: str = "world"


def goal_tag(msg: PoseArray) -> str:
    """Identify a goal by its header stamp, e.g. ``[1700000000.000000123]``."""
    return f"[{msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}]"


def pose_array_to_goals(msg: PoseArray, tool_frames: List[str]) -> List[Dict[str, List[float]]]:
    """Convert a goal PoseArray into goals of ``{tool_frame: [x, y, z, qw, qx, qy, qz]}``."""
    n_frames = len(tool_frames)
    if not msg.poses or len(msg.poses) % n_frames != 0:
        raise ValueError(
            f"expected a multiple of {n_frames} poses ({', '.join(tool_frames)} per goal), "
            f"got {len(msg.poses)}"
        )
    if msg.header.frame_id not in ("", _PLANNING_FRAME):
        raise ValueError(f"poses must be in '{_PLANNING_FRAME}', got '{msg.header.frame_id}'")

    goals = []
    for g in range(len(msg.poses) // n_frames):
        goal = {}
        for f, frame in enumerate(tool_frames):
            p = msg.poses[g * n_frames + f]
            # ROS quaternions are xyzw; cuRobo uses wxyz.
            goal[frame] = [
                p.position.x, p.position.y, p.position.z,
                p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z,
            ]
        goals.append(goal)
    return goals


class PlannerServer:
    """Runs planning and execution on a worker thread while the main thread spins ROS."""

    def __init__(self, node: CuRoboPlanner, robot_file: str, speed: float, dry_run: bool):
        self._node = node
        self._robot_file = robot_file
        self._speed = speed
        self._dry_run = dry_run
        self._planner = None
        # Holds at most one goal; goals arriving while busy are rejected.
        self._goals: "queue.Queue[PoseArray]" = queue.Queue(maxsize=1)
        self._busy = threading.Lock()
        self._status_pub = node.create_publisher(String, STATUS_TOPIC, 10)
        node.create_subscription(PoseArray, GOAL_TOPIC, self._on_goal, 1)
        threading.Thread(target=self._worker, daemon=True).start()

    def _publish_status(self, status: str, goal: Optional[PoseArray] = None) -> None:
        if goal is not None:
            status = f"{goal_tag(goal)} {status}"
        self._node.get_logger().info(f"Status: {status}")
        self._status_pub.publish(String(data=status))

    def _on_goal(self, msg: PoseArray) -> None:
        if self._planner is None:
            self._publish_status("rejected: planner still warming up", msg)
            return
        if not self._busy.acquire(blocking=False):
            self._publish_status("rejected: busy with previous goal", msg)
            return
        self._goals.put(msg)

    def _worker(self) -> None:
        # Warm up on this thread so planning reuses the CUDA graphs captured here.
        planner = MotionPlanner(
            MotionPlannerCfg.create(robot=self._robot_file, scene_model=None)
        )
        self._node.get_logger().info("Warming up motion planner...")
        planner.warmup(enable_graph=True, num_warmup_iterations=5)
        arm_joints = [j for joints in ARM_CONTROLLERS.values() for j in joints]
        missing = [j for j in arm_joints if j not in planner.joint_names]
        if missing:
            self._publish_status(f"failed: {self._robot_file} has no joints named {missing}")
            return
        self._planner = planner
        self._publish_status("ready")
        self._node.get_logger().info(f"Waiting for goals on {GOAL_TOPIC}")

        while True:
            msg = self._goals.get()
            try:
                success, reason = self._plan_and_execute(msg, arm_joints)
                self._publish_status("succeeded" if success else f"failed: {reason}", msg)
            except Exception as e:  # keep the server alive for the next goal
                self._publish_status(f"failed: {e}", msg)
            finally:
                self._busy.release()

    def _plan_and_execute(self, msg: PoseArray, arm_joints: List[str]) -> "tuple[bool, str]":
        planner = self._planner
        goals = pose_array_to_goals(msg, planner.tool_frames)

        live = self._node.wait_for_joint_positions(arm_joints)
        if live is None:
            return False, f"no {JOINT_STATE_TOPIC}"

        # Joints not reported on /joint_states (e.g. grippers) keep their default value.
        start = planner.default_joint_state.position.clone().view(1, -1)
        for i, name in enumerate(planner.joint_names):
            if name in live:
                start[0, i] = live[name]
        start_state = JointState.from_position(start, joint_names=planner.joint_names)

        self._publish_status("planning", msg)
        goal_poses = goals_to_goal_tool_pose(goals, planner.tool_frames, start.device)
        out = multi_phase_planning(planner, start_state, goal_poses, speed=self._speed)
        if out is None:
            return False, "planning failed"
        positions, velocities, dt = out

        if self._dry_run:
            return True, ""
        self._publish_status("executing", msg)
        if not self._node.execute(positions, velocities, dt, planner.joint_names):
            return False, "execution failed"
        return True, ""


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="cuRobo dual-arm planning server")
    parser.add_argument(
        "--robot",
        type=str,
        default="dual_arm.yml",
        help="Robot config file (default: dual_arm.yml)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=DEFAULT_SPEED,
        help=f"Fraction of planned speed, in (0, 1] (default: {DEFAULT_SPEED})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan for each goal but do not send trajectories to the controllers",
    )
    args = parser.parse_args()
    if not 0.0 < args.speed <= 1.0:
        parser.error(f"--speed must be in (0, 1], got {args.speed}")

    rclpy.init()
    node = CuRoboPlanner()
    PlannerServer(node, robot_file=args.robot, speed=args.speed, dry_run=args.dry_run)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
