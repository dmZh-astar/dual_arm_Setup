# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Publish one goal to the planner server and print its status until the motion ends.

    python scripts/send_goal.py                       # built-in MULTI_PHASE_GOALS
    python scripts/send_goal.py --goal-poses goals.yml
"""
import argparse
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from std_msgs.msg import String

from dual_arm_planning import load_goals
from planner_server import GOAL_TOPIC, STATUS_TOPIC, goal_tag

# Tool frames in the order the server expects (planner.tool_frames for dual_arm.yml).
_TOOL_FRAMES = ["left_tcp", "right_tcp"]
_FINAL_STATUSES = ("succeeded", "failed", "rejected")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Send goals to the cuRobo planner server")
    parser.add_argument(
        "--goal-poses",
        type=Path,
        default=None,
        help="YAML list of goals, same structure as MULTI_PHASE_GOALS (default: built-in goals)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for a final status (default: 120)",
    )
    args = parser.parse_args()

    msg = PoseArray()
    msg.header.frame_id = "world"
    for goal in load_goals(args.goal_poses):
        for frame in _TOOL_FRAMES:
            x, y, z, qw, qx, qy, qz = goal[frame]
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = x, y, z
            pose.orientation.w, pose.orientation.x = qw, qx
            pose.orientation.y, pose.orientation.z = qy, qz
            msg.poses.append(pose)

    rclpy.init()
    node = Node("curobo_goal_sender")
    final = []

    def on_status(status: String) -> None:
        tag = goal_tag(msg)
        if not status.data.startswith(tag):
            return  # another sender's goal
        text = status.data[len(tag):].strip()
        print(f"status: {text}")
        if text.startswith(_FINAL_STATUSES):
            final.append(text)

    node.create_subscription(String, STATUS_TOPIC, on_status, 10)
    pub = node.create_publisher(PoseArray, GOAL_TOPIC, 1)

    # Wait for the server to subscribe so the goal is not dropped.
    deadline = time.monotonic() + 10.0
    while pub.get_subscription_count() == 0 and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if pub.get_subscription_count() == 0:
        print(f"No subscriber on {GOAL_TOPIC}; is planner_server.py running?")
        raise SystemExit(1)

    msg.header.stamp = node.get_clock().now().to_msg()
    pub.publish(msg)
    print(f"Sent {len(msg.poses) // len(_TOOL_FRAMES)} goals")

    deadline = time.monotonic() + args.timeout
    while not final and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()
    raise SystemExit(0 if final and final[0] == "succeeded" else 1)


if __name__ == "__main__":
    main()
