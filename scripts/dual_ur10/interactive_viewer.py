# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Interactive Viser viewer for planning with cuRobo. No ROS dependency.

    python scripts/interactive_viewer.py --robot dual_arm.yml --port 8080
"""
import argparse
import time

import numpy as np
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import ContentPath, GoalToolPose, JointState, Pose
from curobo.viewer import ViserVisualizer


def interactive_motion_planning(robot_file="dual_arm.yml", scene_file=None, port=8080):
    """Launch an interactive Viser viewer for motion planning.

    Provides a web-based 3D viewer where you can:
    - Drag the target frame to set the goal pose
    - Drag obstacles to reposition them
    - Click "Move" to plan and execute a collision-free trajectory
    - Click "Grasp" to plan a three-phase grasp motion (approach, grasp, lift)
    """
    import threading

    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file=robot_file),
        connect_ip="0.0.0.0",
        connect_port=port,
        add_control_frames=True,
        visualize_robot_spheres=False,
    )

    config = MotionPlannerCfg.create(robot=robot_file, scene_model=scene_file)
    planner = MotionPlanner(config)

    obstacle_frames = {}
    if config.scene_collision_cfg is not None:
        scene_cfg = config.scene_collision_cfg.scene_model
        obstacle_frames = viser_viz.add_scene(scene_cfg, add_control_frames=True)
    old_obstacle_poses = {
        k: Pose.from_numpy(obstacle_frames[k].position, obstacle_frames[k].wxyz)
        for k in obstacle_frames.keys()
    }

    current_state = planner.default_joint_state.clone().unsqueeze(0)

    print("Warming up motion planner...")
    planner.warmup(enable_graph=True, num_warmup_iterations=5)

    is_moving = False

    def _create_trajectory_image(trajectory, joint_names, title=""):
        """Render a joint trajectory as a PNG image array for the Viser GUI."""
        import io

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        traj = trajectory.squeeze(0)
        pos = np.atleast_2d(traj.position[0].cpu().numpy())  # (horizon, dof)
        dt_val = traj.dt.item() if traj.dt is not None else 0.02
        t = np.arange(pos.shape[0]) * dt_val

        vel = np.atleast_2d(traj.velocity[0].cpu().numpy()) if traj.velocity is not None else None
        acc = np.atleast_2d(traj.acceleration[0].cpu().numpy()) if traj.acceleration is not None else None
        jrk = np.atleast_2d(traj.jerk[0].cpu().numpy()) if traj.jerk is not None else None

        n_plots = 1 + (vel is not None) + (acc is not None) + (jrk is not None)
        fig, axes = plt.subplots(n_plots, 1, figsize=(5, 2 * n_plots), dpi=100, sharex=True)
        if n_plots == 1:
            axes = [axes]

        plot_data = [(pos, "Position (rad)")]
        if vel is not None:
            plot_data.append((vel, "Velocity (rad/s)"))
        if acc is not None:
            plot_data.append((acc, "Accel (rad/s²)"))
        if jrk is not None:
            plot_data.append((jrk, "Jerk (rad/s³)"))

        for ax, (data, ylabel) in zip(axes, plot_data):
            for j in range(data.shape[1]):
                label = joint_names[j] if j < len(joint_names) else f"J{j}"
                if len(label) > 8:
                    label = label[:6] + ".."
                ax.plot(t, data[:, j], linewidth=1.0, label=label)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

        axes[0].legend(loc="upper right", fontsize=7, ncol=2)
        axes[-1].set_xlabel("Time (s)", fontsize=9)
        if title:
            fig.suptitle(title, fontsize=11, fontweight="bold")
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        from PIL import Image
        img_array = np.array(Image.open(buf))
        plt.close(fig)
        buf.close()
        return img_array

    server = viser_viz._server
    traj_plot = server.gui.add_image(
        _create_trajectory_image(
            JointState.from_position(
                planner.default_joint_state.position.unsqueeze(0).unsqueeze(0),
                joint_names=planner.joint_names,
            ),
            planner.joint_names,
            title="No trajectory yet",
        ),
        label="Joint Trajectory",
        format="png",
    )

    def update_obstacles():
        for k in obstacle_frames.keys():
            new_pose = Pose.from_numpy(obstacle_frames[k].position, obstacle_frames[k].wxyz)
            if new_pose != old_obstacle_poses[k]:
                planner.scene_collision_checker.update_obstacle_pose(k, new_pose)
                old_obstacle_poses[k] = new_pose.clone()

    def print_tcp_positions(joint_state: JointState) -> None:
        """Print every tool frame pose for *joint_state* in the world frame (qw, qx, qy, qz)."""
        kin_state = planner.compute_kinematics(planner.kinematics.get_active_js(joint_state))
        positions = kin_state.tool_poses.position.view(-1, 3).cpu().numpy()
        quaternions = kin_state.tool_poses.quaternion.view(-1, 4).cpu().numpy()
        for name, pos, quat in zip(planner.tool_frames, positions, quaternions):
            print(
                f"  {name}: x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}, "
                f"qw={quat[0]:.4f}, qx={quat[1]:.4f}, qy={quat[2]:.4f}, qz={quat[3]:.4f}"
            )

    def execute_trajectory(trajectory):
        nonlocal current_state, is_moving
        traj = trajectory.squeeze(0)
        for i in range(traj.position.shape[-2]):
            if not is_moving:
                return
            waypoint = JointState.from_position(
                traj.position[0, i, :].unsqueeze(0),
                joint_names=traj.joint_names,
            )
            viser_viz.set_joint_state(waypoint.squeeze(0))
            time.sleep(0.02)
        current_state = JointState.from_position(
            traj.position[0, -1, :].unsqueeze(0),
            joint_names=traj.joint_names,
        )
        print("TCP poses after move:")
        print_tcp_positions(current_state)

    def on_move(_):
        nonlocal is_moving
        if is_moving:
            return

        def plan_and_execute():
            nonlocal is_moving
            is_moving = True
            update_obstacles()
            target_poses = viser_viz.get_control_frame_pose()
            active_js = planner.kinematics.get_active_js(current_state.clone())
            result = planner.plan_pose(
                GoalToolPose.from_poses(target_poses, num_goalset=1),
                active_js,
                use_implicit_goal=True,
                max_attempts=3,
            )
            if result is not None and result.success.any():
                interp = result.get_interpolated_plan()
                traj_plot.image = _create_trajectory_image(
                    interp, planner.joint_names,
                    title=f"Pose Plan  |  {result.total_time:.3f}s",
                )
                execute_trajectory(interp)
            else:
                print("Motion planning failed")
            is_moving = False

        threading.Thread(target=plan_and_execute, daemon=True).start()

    def on_grasp(_):
        nonlocal is_moving
        if is_moving:
            return

        def plan_grasp_and_execute():
            nonlocal is_moving
            is_moving = True
            update_obstacles()
            target_poses = viser_viz.get_control_frame_pose()
            active_js = planner.kinematics.get_active_js(current_state.clone())
            grasp_poses = GoalToolPose.from_poses(target_poses, num_goalset=1)
            result = planner.plan_grasp(
                grasp_poses,
                active_js,
                plan_approach_to_grasp=True,
                plan_grasp_to_lift=True,
                grasp_lift_in_tool_frame=True,
            )
            if result is not None and result.success is not None and result.success.any():
                traj_plot.image = _create_trajectory_image(
                    result.approach_interpolated_trajectory, planner.joint_names,
                    title="Approach",
                )
                execute_trajectory(result.approach_interpolated_trajectory)
                traj_plot.image = _create_trajectory_image(
                    result.grasp_interpolated_trajectory, planner.joint_names,
                    title="Grasp",
                )
                execute_trajectory(result.grasp_interpolated_trajectory)
                traj_plot.image = _create_trajectory_image(
                    result.lift_interpolated_trajectory, planner.joint_names,
                    title="Lift",
                )
                execute_trajectory(result.lift_interpolated_trajectory)
            else:
                print("Grasp planning failed")
            is_moving = False

        threading.Thread(target=plan_grasp_and_execute, daemon=True).start()

    move_btn = server.gui.add_button("Move", color="green")
    move_btn.on_click(on_move)
    grasp_btn = server.gui.add_button("Grasp", color="blue")
    grasp_btn.on_click(on_grasp)

    print(f"\nInteractive Motion Planner running at http://localhost:{port}")
    print("  - Drag the target frame to set goal pose")
    print("  - Drag obstacles to reposition them")
    print("  - Click 'Move' for pose-to-pose planning")
    print("  - Click 'Grasp' for approach-grasp-lift planning")
    print("Press Ctrl+C to exit.\n")
    print("Initial TCP poses:")
    print_tcp_positions(current_state)

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Interactive motion planning with cuRobo")
    parser.add_argument(
        "--robot",
        type=str,
        default="dual_arm.yml",
        help="Robot config file (default: dual_arm.yml)",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Scene config file (default: None)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Viser server port (default: 8080)",
    )
    args = parser.parse_args()
    interactive_motion_planning(robot_file=args.robot, scene_file=args.scene, port=args.port)


if __name__ == "__main__":
    main()
