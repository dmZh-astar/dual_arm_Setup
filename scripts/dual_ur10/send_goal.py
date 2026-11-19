# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Publish one goal to the planner server and print its status until the motion ends.

Goal poses are given in ``right_base`` (the right arm's UR Base frame) and converted to the
cuRobo ``world`` frame before sending. On success, the reached TCP poses are printed in
``right_base``.

    python scripts/send_goal.py                       # goals.yml next to this script
    python scripts/send_goal.py --goal-poses goals.yml
"""
import argparse
import math
import time
from pathlib import Path
from typing import List

import rclpy
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from dual_arm_planning import load_goals
from planner_server import GOAL_TOPIC, STATUS_TOPIC, goal_tag

# Tool frames in the order the server expects (planner.tool_frames for dual_arm.yml).
_TOOL_FRAMES = ["left_tcp", "right_tcp"]
_FINAL_STATUSES = ("succeeded", "failed", "rejected")


def right_base_to_world(pose: List[float]) -> List[float]:
    """Convert ``[x, y, z, qw, qx, qy, qz]`` from right_base to world.

    right_base sits at (0.6, -0.15, 0.8) in world, rotated -90 deg about Z (from the URDF).
    """
    x, y, z, qw, qx, qy, qz = pose
    c = math.sqrt(0.5)
    return [
        0.6 + y, -0.15 - x, 0.8 + z,
        c * (qw + qz), c * (qx + qy), c * (qy - qx), c * (qz - qw),
    ]


def world_to_right_base(pose: List[float]) -> List[float]:
    """Convert ``[x, y, z, qw, qx, qy, qz]`` from world to right_base (inverse of the above).

    The quaternion is returned with qw >= 0: q and -q are the same rotation, and TF may give
    either sign.
    """
    x, y, z, qw, qx, qy, qz = pose
    c = math.sqrt(0.5)
    q = [c * (qw - qz), c * (qx - qy), c * (qy + qx), c * (qz + qw)]
    if q[0] < 0:
        q = [-v for v in q]
    return [-0.15 - y, x - 0.6, z - 0.8] + q


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Send goals to the cuRobo planner server")
    parser.add_argument(
        "--goal-poses",
        type=Path,
        default=Path(__file__).parent / "goals.yml",
        help="YAML list of goals in right_base (default: goals.yml next to this script)",
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
            x, y, z, qw, qx, qy, qz = right_base_to_world(goal[frame])
            # x, y, z, qw, qx, qy, qz = goal[frame]
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = x, y, z
            pose.orientation.w, pose.orientation.x = qw, qx
            pose.orientation.y, pose.orientation.z = qy, qz
            msg.poses.append(pose)

    rclpy.init()
    node = Node("curobo_goal_sender")
    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)
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

    succeeded = bool(final) and final[0] == "succeeded"
    if succeeded:
        # Spin briefly so TF holds the final robot state, then print the reached poses.
        settle = time.monotonic() + 0.5
        while time.monotonic() < settle:
            rclpy.spin_once(node, timeout_sec=0.1)
        for frame in _TOOL_FRAMES:
            t = tf_buffer.lookup_transform("world", frame, rclpy.time.Time()).transform
            p, q = t.translation, t.rotation
            values = world_to_right_base([p.x, p.y, p.z, q.w, q.x, q.y, q.z])
            # values = [p.x, p.y, p.z, q.w, q.x, q.y, q.z]
            print(f"reached {frame} in right_base: {[round(v, 4) for v in values]}")

    node.destroy_node()
    rclpy.shutdown()
    raise SystemExit(0 if succeeded else 1)


if __name__ == "__main__":
    main()
