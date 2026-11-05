# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Dual-arm multi-goal planning with cuRobo. No ROS dependency."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from curobo.motion_planner import MotionPlanner
from curobo.types import GoalToolPose, JointState
# Default fraction of the planned speed sent to the real robot.
DEFAULT_SPEED: float = 0.1

# Goals used when --goal-poses is not given, visited in order. Each goal gives one pose per
# tool frame as [x, y, z, qw, qx, qy, qz] in the cuRobo world frame. Keep an arm still by
# repeating its pose. A --goal-poses YAML file uses the same structure.
MULTI_PHASE_GOALS = [
    {
        "left_tcp": [-0.1124, 0.2118, 1.2479, 0.4716, -0.5741, 0.5579, -0.3698],
        "right_tcp": [0.0386, -0.0845, 1.0578, 0.6451, -0.1423, -0.7181, -0.2189],
    },
    {
        "left_tcp": [-0.1124, 0.2118, 1.2479, 0.4716, -0.5741, 0.5579, -0.3698],
        "right_tcp": [-0.1131, 0.1230, 1.1040, 0.6451, -0.1423, -0.7181, -0.2189],
    },
]



def load_goals(path: Optional[Path]) -> List[Dict[str, List[float]]]:
    """Load goals from a YAML file, or return ``MULTI_PHASE_GOALS`` when *path* is None."""
    if path is None:
        return MULTI_PHASE_GOALS
    import yaml

    with open(path) as f:
        goals = yaml.safe_load(f)
    if not isinstance(goals, list) or not goals:
        raise ValueError(f"{path} must contain a non-empty list of goals")
    return goals


def goals_to_goal_tool_pose(
    goals: List[Dict[str, List[float]]], tool_frames: List[str], device: torch.device
) -> GoalToolPose:
    """Stack goals into a GoalToolPose of shape ``(1, 1, n_tool_frames, n_goals, 3|4)``."""
    for i, goal in enumerate(goals):
        missing = [f for f in tool_frames if f not in goal]
        if missing:
            raise ValueError(f"Goal {i + 1} is missing poses for {missing}")
        bad = [f for f in tool_frames if len(goal[f]) != 7]
        if bad:
            raise ValueError(f"Goal {i + 1}: poses for {bad} must be [x, y, z, qw, qx, qy, qz]")

    poses = torch.tensor(
        [[goal[frame] for goal in goals] for frame in tool_frames],
        device=device,
        dtype=torch.float32,
    )
    return GoalToolPose(
        tool_frames=tool_frames,
        position=poses[..., :3].unsqueeze(0).unsqueeze(0).contiguous(),
        quaternion=poses[..., 3:].unsqueeze(0).unsqueeze(0).contiguous(),
    )


def multi_phase_planning(
    planner: MotionPlanner,
    start_state: JointState,
    goal_poses: GoalToolPose,
    speed: float = DEFAULT_SPEED,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Plan through several goal poses in order and join them into one trajectory.

    Goal ``i`` is planned from the last waypoint of goal ``i - 1``, so every segment is
    collision-checked and the joins are exact. Each segment ends at rest, so the robot
    briefly stops at every intermediate goal.

    Args:
        planner: A warmed-up motion planner.
        start_state: Start joint state, shape ``(1, dof)``.
        goal_poses: Goals in visiting order along the goalset dimension, position shape
            ``(1, 1, n_tool_frames, n_goals, 3)`` and quaternion ``(1, 1, n_tool_frames,
            n_goals, 4)`` (wxyz).
        speed: Fraction of the planned speed in ``(0, 1]``. Only stretches time; the path
            is unchanged, so it stays collision-free.

    Returns:
        ``(positions, velocities, dt)``: arrays of shape ``(n_waypoints, dof)`` in
        ``planner.joint_names`` order and the waypoint spacing in seconds, or None if any
        goal could not be reached.
    """
    if not 0.0 < speed <= 1.0:
        raise ValueError(f"speed must be in (0, 1], got {speed}")

    n_goals = goal_poses.position.shape[3]
    print(f"Planning through {n_goals} goal poses at speed {speed:.2f}...")

    positions = []
    velocities = []
    for i in range(n_goals):
        goal = GoalToolPose(
            tool_frames=goal_poses.tool_frames,
            position=goal_poses.position[:, :, :, i : i + 1, :],
            quaternion=goal_poses.quaternion[:, :, :, i : i + 1, :],
        )
        result = planner.plan_pose(goal, planner.kinematics.get_active_js(start_state))
        if result is None or not result.success.any():
            status = getattr(result, "status", "unknown")
            print(f"Planning failed at goal {i + 1}: {status}")
            return None

        # get_interpolated_plan() drops the unused padding at the end of the buffer.
        interp = result.get_interpolated_plan()
        pos = interp.position.reshape(-1, planner.action_dim)
        vel = interp.velocity.reshape(-1, planner.action_dim)
        print(f"  Goal {i + 1}: {pos.shape[0]} waypoints, planned in {float(result.total_time):.3f}s")
        # The first waypoint repeats the previous segment's end.
        positions.append(pos if i == 0 else pos[1:])
        velocities.append(vel if i == 0 else vel[1:])
        start_state = JointState.from_position(pos[-1:].clone(), joint_names=planner.joint_names)

    positions = torch.cat(positions, dim=0).cpu().numpy()
    # Stretching time by 1 / speed scales velocities by speed.
    velocities = (torch.cat(velocities, dim=0) * speed).cpu().numpy()
    dt = planner.trajopt_solver.config.interpolation_dt / speed
    print(f"Planning succeeded: {positions.shape[0]} waypoints, "
          f"{positions.shape[0] * dt:.2f}s at dt={dt:.4f}s")
    return positions, velocities, dt
