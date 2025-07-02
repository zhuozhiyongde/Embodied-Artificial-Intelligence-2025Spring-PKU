import argparse
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image
import tqdm

# 添加Assignment2/src到路径，以便导入constants
# sys.path.append("Assignment2/src")
from src.constants import PC_MAX, PC_MIN
from src.robot.cfg import WRIST_CAMERA
from src.sim.wrapper_env import WrapperEnv, WrapperEnvConfig
from src.utils import get_pc
from src.vis import Vis

# --- Default Configuration ---
DEFAULT_NUM_SAMPLES = 5000
DEFAULT_OUTPUT_DIR = "data"
DEFAULT_OBJECT_NAME = "power_drill"
DEFAULT_TRAIN_RATIO = 0.8


def parse_args():
    parser = argparse.ArgumentParser(description="生成电钻位姿数据集")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f"生成的样本数量 (默认: {DEFAULT_NUM_SAMPLES})",
    )
    parser.add_argument(
        "--range",
        type=str,
        default=None,
        help="生成样本的范围，格式为 'start,end' (左闭右开区间)，会覆盖 num_samples 参数",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录 (默认: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--object_name",
        type=str,
        default=DEFAULT_OBJECT_NAME,
        help=f"物体名称 (默认: {DEFAULT_OBJECT_NAME})",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help=f"训练集比例 (默认: {DEFAULT_TRAIN_RATIO})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细模式，启用所有打印输出",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="可视化模式，显示电钻点云的边界框范围",
    )
    parser.add_argument(
        "--individual",
        action="store_true",
        help="为每个样本单独保存可视化文件",
    )
    parser.add_argument(
        "--save_pc",
        action="store_true",
        default=False,
        help="保存点云数据",
    )
    parser.add_argument(
        "--vis_mode",
        type=str,
        choices=["obj", "all"],
        default="obj",
        help="可视化模式：obj=只显示物体点云，all=显示所有点云 (默认: obj)",
    )
    return parser.parse_args()


def execute_plan(env: WrapperEnv, plan):
    """Execute the plan in the environment."""
    for step in range(len(plan)):
        env.step_env(humanoid_action=plan[step])


def plan_move_qpos(env: WrapperEnv, begin_qpos, end_qpos, steps=50) -> np.ndarray:
    delta_qpos = (end_qpos - begin_qpos) / steps
    cur_qpos = begin_qpos.copy()
    traj = []

    for _ in range(steps):
        cur_qpos += delta_qpos
        traj.append(cur_qpos.copy())

    return np.array(traj)


def save_sample_data(sample_folder, obs, env: WrapperEnv, verbose=False, save_pc=False):
    """Save data using debug_save_obs format with additional object segmentation"""

    # 首先使用debug_save_obs来保存基础数据
    env.debug_save_obs(obs, sample_folder)

    # 保存物体位姿
    # object_pose = data_dict["obj_pose"]
    object_pose = env.get_driller_pose()
    np.save(os.path.join(sample_folder, "object_pose.npy"), object_pose)

    # 保存电钻分割掩码（保持现有的分割方式）
    if obs.seg is not None:
        # 直接使用从仿真中确定的电钻分割ID
        # 电钻由多个几何体组成，ID范围为106-119（power_drill_collision_0到power_drill_collision_13）
        DRILL_SEGMENTATION_IDS = list(range(106, 120))  # 106, 107, 108, ..., 119

        # 创建电钻的二值掩码
        drill_mask = np.zeros_like(obs.seg, dtype=np.uint8)

        drill_pixel_count = 0
        for drill_id in DRILL_SEGMENTATION_IDS:
            id_mask = obs.seg == drill_id
            drill_mask |= id_mask
            drill_pixel_count += np.sum(id_mask)

        # 转换为二值掩码 (0和255)
        drill_mask = drill_mask.astype(np.uint8) * 255

        # 计算电钻占图像的百分比
        drill_percentage = (drill_pixel_count / obs.seg.size) * 100

        if verbose:
            print(f"  -> 电钻分割: {drill_pixel_count} 像素 ({drill_percentage:.2f}%)")

        cv2.imwrite(os.path.join(sample_folder, "obj_seg.png"), drill_mask)

    if save_pc:
        # 保存分割出的电钻点云为npy文件
        try:
            driller_points_camera = get_driller_pointcloud_from_obs(obs)
            if len(driller_points_camera) > 0:
                # 保存相机坐标系下的电钻点云
                np.save(
                    os.path.join(sample_folder, "driller_pc_camera.npy"),
                    driller_points_camera,
                )

                # 转换到世界坐标系并保存
                points_homo = np.hstack(
                    [driller_points_camera, np.ones((len(driller_points_camera), 1))]
                )
                driller_points_world = (obs.camera_pose @ points_homo.T).T[:, :3]
                np.save(
                    os.path.join(sample_folder, "driller_pc_world.npy"),
                    driller_points_world,
                )

                if verbose:
                    print(
                        f"  -> 电钻点云: {len(driller_points_camera)} 个点 (相机坐标系和世界坐标系)"
                    )
            else:
                if verbose:
                    print(f"  -> 警告: 未找到有效的电钻点云")
        except Exception as e:
            if verbose:
                print(f"  -> 保存电钻点云时出错: {e}")

        # 保存完整场景点云
        try:
            full_pc_camera, full_pc_world = get_pc_from_obs(obs)
            np.save(
                os.path.join(sample_folder, "full_pc_camera_raw.npy"), full_pc_camera
            )
            np.save(os.path.join(sample_folder, "full_pc_world_raw.npy"), full_pc_world)

            if verbose:
                print(
                    f"  -> 完整点云: {len(full_pc_camera)} 个点 (相机坐标系和世界坐标系)"
                )
        except Exception as e:
            if verbose:
                print(f"  -> 保存完整点云时出错: {e}")


def get_pc_from_obs(obs):
    """从观测数据中计算点云，使用类似data.py的方式"""

    # 使用与data.py相同的方式计算点云
    depth_array = obs.depth
    intrinsics = WRIST_CAMERA.intrinsics

    full_pc_camera = get_pc(depth_array, intrinsics)

    # 将点云转换到世界坐标系
    camera_pose = obs.camera_pose
    full_pc_world = (
        np.einsum("ab,nb->na", camera_pose[:3, :3], full_pc_camera) + camera_pose[:3, 3]
    )

    return full_pc_camera, full_pc_world


def get_driller_pointcloud_from_obs(obs):
    """从观测数据中提取电钻点云，使用类似data.py的点云计算方式"""
    # 电钻分割ID范围
    DRILL_SEGMENTATION_IDS = list(range(106, 120))  # 106, 107, 108, ..., 119

    # 创建电钻的二值掩码
    drill_mask = np.zeros_like(obs.seg, dtype=bool)
    for drill_id in DRILL_SEGMENTATION_IDS:
        drill_mask |= obs.seg == drill_id

    # 使用类似data.py的方式计算点云
    full_pc_camera, full_pc_world = get_pc_from_obs(obs)

    # 应用电钻掩码
    driller_points_camera_frame = full_pc_camera[drill_mask.flatten()]

    # 过滤无效点 (深度为0或过大的点)
    valid_mask = (driller_points_camera_frame[:, 2] > 0) & (
        driller_points_camera_frame[:, 2] < 3.0
    )
    driller_points_camera_frame = driller_points_camera_frame[valid_mask]

    return driller_points_camera_frame


def generate_data(args):
    # 处理 range 参数
    if args.range is not None:
        try:
            start_idx, end_idx = map(int, args.range.split(","))
            NUM_SAMPLES = end_idx - start_idx
            SAMPLE_START_IDX = start_idx
        except ValueError:
            raise ValueError(
                "--range 参数格式错误，应为 'start,end' 格式，例如 '0,1000'"
            )
    else:
        NUM_SAMPLES = args.num_samples
        SAMPLE_START_IDX = 0

    OUTPUT_DIR = args.output_dir
    OBJECT_NAME = args.object_name
    TRAIN_RATIO = args.train_ratio
    verbose = args.verbose
    save_pc = args.save_pc

    # 用于收集每个样本的电钻点云边界值进行统计分析
    sample_bounds_world = []  # 存储每个样本世界坐标系下的边界值 [min_x, min_y, min_z, max_x, max_y, max_z]
    sample_bounds_camera = []  # 存储每个样本相机坐标系下的边界值

    # 用于收集所有样本的可视化数据（如果需要综合可视化）
    all_vis_data = []  # 存储每个样本的可视化数据 {driller_points, camera_pose, object_pose, sample_id}

    if verbose:
        if args.range is not None:
            print(
                f"开始生成样本范围 [{SAMPLE_START_IDX}, {SAMPLE_START_IDX + NUM_SAMPLES}) 的数据..."
            )
        else:
            print(f"开始生成 {NUM_SAMPLES} 个样本的数据...")

    # Create directory structure
    train_dir = os.path.join(OUTPUT_DIR, OBJECT_NAME, "train")
    val_dir = os.path.join(OUTPUT_DIR, OBJECT_NAME, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Calculate split based on global sample indices
    # 使用全局样本索引来确定训练/验证集分配，保证一致性
    global_train_samples = int(args.num_samples * TRAIN_RATIO)  # 假设总体有5000个样本

    if verbose:
        print(f"样本范围: [{SAMPLE_START_IDX}, {SAMPLE_START_IDX + NUM_SAMPLES})")
        print(f"全局训练样本数: {global_train_samples}")

    train_metadata = []
    val_metadata = []

    for i in tqdm.tqdm(range(NUM_SAMPLES), desc="Generating data"):
        # 计算全局样本ID
        global_sample_id = SAMPLE_START_IDX + i

        env_config = WrapperEnvConfig(
            humanoid_robot="galbot",
            obj_name=OBJECT_NAME,
            headless=1,
        )
        env = WrapperEnv(env_config)

        env.launch()

        env.reset(humanoid_qpos=env.sim.humanoid_robot_cfg.joint_init_qpos)
        humanoid_init_qpos = env.sim.humanoid_robot_cfg.joint_init_qpos[:7]

        head_init_qpos = np.array([-0.05, 0.35])  # [horizontal, vertical]
        env.step_env(humanoid_head_qpos=head_init_qpos)

        observing_qpos = humanoid_init_qpos + np.array([0.01, 0, 0.25, 0, 0, 0, 0.15])
        init_plan = plan_move_qpos(env, humanoid_init_qpos, observing_qpos, steps=20)
        execute_plan(env, init_plan)

        # Get observation from wrist camera (camera_id=1)
        obs = env.get_obs(camera_id=1)

        if obs is None or obs.depth is None or obs.seg is None or obs.rgb is None:
            if verbose:
                print(f"警告: 无法获取样本 {global_sample_id} 的有效观测数据。跳过。")
            continue

        # Determine if this goes to train or val based on global sample ID
        is_train = global_sample_id < global_train_samples
        base_dir = train_dir if is_train else val_dir

        # Create sample folder using global sample ID
        sample_folder = os.path.join(base_dir, f"sample_{global_sample_id:06d}")
        os.makedirs(sample_folder, exist_ok=True)

        # Save data using debug_save_obs format
        save_sample_data(sample_folder, obs, env, verbose=verbose, save_pc=save_pc)

        # 收集电钻点云边界值用于统计分析
        try:
            driller_points_camera = get_driller_pointcloud_from_obs(obs)
            if len(driller_points_camera) > 0:
                # 计算相机坐标系下的边界值
                min_camera = np.min(driller_points_camera, axis=0)
                max_camera = np.max(driller_points_camera, axis=0)
                sample_bounds_camera.append(np.concatenate([min_camera, max_camera]))

                # 转换到世界坐标系并计算边界值
                points_homo = np.hstack(
                    [driller_points_camera, np.ones((len(driller_points_camera), 1))]
                )
                driller_points_world = (obs.camera_pose @ points_homo.T).T[:, :3]
                min_world = np.min(driller_points_world, axis=0)
                max_world = np.max(driller_points_world, axis=0)
                sample_bounds_world.append(np.concatenate([min_world, max_world]))

                if verbose:
                    print(
                        f"  -> 样本 {global_sample_id} 电钻边界: X[{min_world[0]:.3f}, {max_world[0]:.3f}], Y[{min_world[1]:.3f}, {max_world[1]:.3f}], Z[{min_world[2]:.3f}, {max_world[2]:.3f}]"
                    )
        except Exception as e:
            if verbose:
                print(f"收集样本 {global_sample_id} 的电钻点云边界值时出错: {e}")

        if args.vis and NUM_SAMPLES > 200:
            args.vis = False
            print("采样数量过多，关闭可视化")

        # 收集可视化数据
        if args.vis:
            try:
                # 提取电钻点云
                driller_points = get_driller_pointcloud_from_obs(obs)

                if args.individual:
                    # 单独可视化每个样本
                    vis_output_dir = os.path.join(args.output_dir, "visualizations")
                    html_path = visualize_driller_bbox(
                        driller_points,
                        obs.camera_pose,
                        env.get_driller_pose(),
                        sample_id=f"sample_{global_sample_id:06d}",
                        output_dir=vis_output_dir,
                        verbose=verbose,
                        vis_mode=args.vis_mode,
                        obs=obs,
                    )
                else:
                    # 收集数据用于综合可视化
                    if len(driller_points) > 0:
                        vis_data = {
                            "driller_points": driller_points,
                            "camera_pose": obs.camera_pose,
                            "object_pose": env.get_driller_pose(),
                            "sample_id": f"sample_{global_sample_id:06d}",
                        }
                        # 如果需要显示所有点云，则保存obs数据
                        if args.vis_mode == "all":
                            vis_data["obs"] = obs
                        all_vis_data.append(vis_data)
            except Exception as e:
                if verbose:
                    print(f"可视化样本 {global_sample_id} 时出错: {e}")

        # Create metadata
        sample_metadata = {
            "sample_id": f"sample_{global_sample_id:06d}",
            "rgb_path": os.path.join(sample_folder, "rgb.png").replace("\\", "/"),
            "depth_path": os.path.join(sample_folder, "depth.png").replace("\\", "/"),
            "obj_seg_path": os.path.join(sample_folder, "obj_seg.png").replace(
                "\\", "/"
            ),
            "object_pose_path": os.path.join(sample_folder, "object_pose.npy").replace(
                "\\", "/"
            ),
            "camera_pose_path": os.path.join(sample_folder, "camera_pose.npy").replace(
                "\\", "/"
            ),
        }

        if is_train:
            train_metadata.append(sample_metadata)
        else:
            val_metadata.append(sample_metadata)

        if (i + 1) % 10 == 0 and verbose:
            print(
                f"已生成 {i + 1}/{NUM_SAMPLES} 个样本... (全局ID: {SAMPLE_START_IDX}~{global_sample_id})"
            )

        env.close()

    # Save metadata files
    with open(os.path.join(train_dir, "metadata.json"), "w") as f:
        json.dump(train_metadata, f, indent=4)

    with open(os.path.join(val_dir, "metadata.json"), "w") as f:
        json.dump(val_metadata, f, indent=4)

    # 生成综合可视化（如果启用了可视化且没有开启individual模式）
    if args.vis and not args.individual and len(all_vis_data) > 0:
        try:
            vis_output_dir = os.path.join(args.output_dir, "visualizations")
            html_path = visualize_all_driller_bbox(
                all_vis_data,
                output_dir=vis_output_dir,
                verbose=verbose,
                vis_mode=args.vis_mode,
            )
            if verbose:
                print(f"综合可视化已保存到: {html_path}")
        except Exception as e:
            if verbose:
                print(f"生成综合可视化时出错: {e}")

    # 统计分析所有电钻点云边界值
    if len(sample_bounds_world) > 0 and len(sample_bounds_camera) > 0:
        analyze_driller_bounds_statistics(
            sample_bounds_world, sample_bounds_camera, verbose=verbose
        )
    elif verbose:
        print("警告: 没有收集到有效的电钻点云边界值进行统计分析")

    if verbose:
        print(f"数据生成完成！")
        print(f"训练集: {len(train_metadata)} 个样本 -> {train_dir}")
        print(f"验证集: {len(val_metadata)} 个样本 -> {val_dir}")

    return train_dir, val_dir


def test_generated_data(train_dir, val_dir, verbose=False):
    """测试生成的数据是否符合目标格式"""
    if verbose:
        print("\n=== 测试生成的数据格式 ===")

    # Try to find sample from train set first, then val set
    sample_dirs = [d for d in os.listdir(train_dir) if d.startswith("sample_")]
    test_dir = train_dir

    if not sample_dirs:
        # If no train samples, try val set
        sample_dirs = [d for d in os.listdir(val_dir) if d.startswith("sample_")]
        test_dir = val_dir

    if not sample_dirs:
        if verbose:
            print("没有找到生成的样本数据")
        return

    sample_path = os.path.join(test_dir, sample_dirs[0])
    if verbose:
        print(f"测试样本: {sample_path}")

    # Check files exist
    required_files = [
        "rgb.png",
        "depth.png",
        "obj_seg.png",
        "object_pose.npy",
        "camera_pose.npy",
    ]
    for file in required_files:
        file_path = os.path.join(sample_path, file)
        if os.path.exists(file_path):
            if verbose:
                print(f"✓ {file} 存在")
        else:
            if verbose:
                print(f"✗ {file} 不存在")

    # Check npy file structures
    try:
        object_pose = np.load(os.path.join(sample_path, "object_pose.npy"))
        camera_pose = np.load(os.path.join(sample_path, "camera_pose.npy"))

        if verbose:
            print(f"object_pose shape: {object_pose.shape}, dtype: {object_pose.dtype}")
            print(f"camera_pose shape: {camera_pose.shape}, dtype: {camera_pose.dtype}")

            # Check images
            rgb = cv2.imread(os.path.join(sample_path, "rgb.png"))
            depth = cv2.imread(
                os.path.join(sample_path, "depth.png"), cv2.IMREAD_UNCHANGED
            )
            seg = cv2.imread(
                os.path.join(sample_path, "obj_seg.png"), cv2.IMREAD_GRAYSCALE
            )

            print(f"RGB shape: {rgb.shape}, dtype: {rgb.dtype}")
            print(f"Depth shape: {depth.shape}, dtype: {depth.dtype}")
            print(f"Segmentation shape: {seg.shape}, dtype: {seg.dtype}")
            print(f"Segmentation unique values: {np.unique(seg)}")

    except Exception as e:
        if verbose:
            print(f"测试数据时出错: {e}")


def analyze_driller_bounds_statistics(
    sample_bounds_world, sample_bounds_camera, verbose=False
):
    """分析所有电钻点云边界值的统计信息"""

    if verbose:
        print("\n" + "=" * 60)
        print("电钻点云边界值统计分析")
        print("=" * 60)

    # 转换为numpy数组以便于计算
    bounds_world = np.array(
        sample_bounds_world
    )  # shape: (num_samples, 6) - [min_x, min_y, min_z, max_x, max_y, max_z]
    bounds_camera = np.array(sample_bounds_camera)

    if verbose:
        print(f"总样本数: {len(sample_bounds_world)}")

    # 世界坐标系下的边界值统计
    print(f"\n世界坐标系下的边界值统计:")
    print("-" * 50)

    dimensions = ["X", "Y", "Z"]
    for dim_idx, dim_name in enumerate(dimensions):
        min_values = bounds_world[:, dim_idx]  # 所有样本的min值
        max_values = bounds_world[:, dim_idx + 3]  # 所有样本的max值

        print(f"\n{dim_name}维度边界统计:")
        print(
            f"  最小边界: 最小={np.min(min_values):.4f}, 最大={np.max(min_values):.4f}, 平均={np.mean(min_values):.4f}"
        )
        print(
            f"  最大边界: 最小={np.min(max_values):.4f}, 最大={np.max(max_values):.4f}, 平均={np.mean(max_values):.4f}"
        )
        print(f"  总体范围: [{np.min(min_values):.4f}, {np.max(max_values):.4f}]")

    # 相机坐标系下的边界值统计
    print(f"\n相机坐标系下的边界值统计:")
    print("-" * 50)

    for dim_idx, dim_name in enumerate(dimensions):
        min_values = bounds_camera[:, dim_idx]  # 所有样本的min值
        max_values = bounds_camera[:, dim_idx + 3]  # 所有样本的max值

        print(f"\n{dim_name}维度边界统计:")
        print(
            f"  最小边界: 最小={np.min(min_values):.4f}, 最大={np.max(min_values):.4f}, 平均={np.mean(min_values):.4f}"
        )
        print(
            f"  最大边界: 最小={np.min(max_values):.4f}, 最大={np.max(max_values):.4f}, 平均={np.mean(max_values):.4f}"
        )
        print(f"  总体范围: [{np.min(min_values):.4f}, {np.max(max_values):.4f}]")

    # 与constants中定义的PC_MIN, PC_MAX进行对比
    print(f"\n与constants.py中PC_MIN/PC_MAX的对比:")
    print("-" * 50)
    print(f"PC_MIN: [{PC_MIN[0]:.3f}, {PC_MIN[1]:.3f}, {PC_MIN[2]:.3f}]")
    print(f"PC_MAX: [{PC_MAX[0]:.3f}, {PC_MAX[1]:.3f}, {PC_MAX[2]:.3f}]")

    # 计算总体实际数据范围
    overall_min = [np.min(bounds_world[:, i]) for i in range(3)]
    overall_max = [np.max(bounds_world[:, i + 3]) for i in range(3)]

    print(f"实际数据总体范围:")
    print(f"  Min: [{overall_min[0]:.3f}, {overall_min[1]:.3f}, {overall_min[2]:.3f}]")
    print(f"  Max: [{overall_max[0]:.3f}, {overall_max[1]:.3f}, {overall_max[2]:.3f}]")

    # 建议的新PC_MIN/PC_MAX（基于总体范围+缓冲）
    buffer = 0.05
    suggested_min = [overall_min[i] - buffer for i in range(3)]
    suggested_max = [overall_max[i] + buffer for i in range(3)]

    print(f"建议的PC范围 (基于总体范围+{buffer}缓冲):")
    print(
        f"  建议PC_MIN: [{suggested_min[0]:.3f}, {suggested_min[1]:.3f}, {suggested_min[2]:.3f}]"
    )
    print(
        f"  建议PC_MAX: [{suggested_max[0]:.3f}, {suggested_max[1]:.3f}, {suggested_max[2]:.3f}]"
    )

    if verbose:
        print("=" * 60)


def visualize_driller_bbox(
    driller_points,
    camera_pose,
    object_pose,
    sample_id="",
    output_dir="vis_output",
    verbose=False,
    vis_mode="obj",
    obs=None,
):
    """可视化电钻点云的边界框"""
    if len(driller_points) == 0:
        if verbose:
            print(f"样本 {sample_id}: 没有找到电钻点云数据")
        return

    # 将点云从相机坐标系转换到世界坐标系
    # 添加齐次坐标
    points_homo = np.hstack([driller_points, np.ones((len(driller_points), 1))])
    # 转换到世界坐标系
    world_points = (camera_pose @ points_homo.T).T[:, :3]

    # 计算边界框
    min_xyz = np.min(world_points, axis=0)
    max_xyz = np.max(world_points, axis=0)
    center_xyz = (min_xyz + max_xyz) / 2
    size_xyz = max_xyz - min_xyz

    # 计算在 PC_MIN 到 PC_MAX 区间内的点的比例
    in_bounds_mask = (
        (world_points[:, 0] >= PC_MIN[0])
        & (world_points[:, 0] <= PC_MAX[0])
        & (world_points[:, 1] >= PC_MIN[1])
        & (world_points[:, 1] <= PC_MAX[1])
        & (world_points[:, 2] >= PC_MIN[2])
        & (world_points[:, 2] <= PC_MAX[2])
    )
    in_bounds_count = np.sum(in_bounds_mask)
    in_bounds_ratio = (
        in_bounds_count / len(world_points) if len(world_points) > 0 else 0.0
    )

    if verbose:
        print(f"样本 {sample_id}: 电钻点云统计")
        print(f"  点数: {len(driller_points)}")
        print(f"  X范围: [{min_xyz[0]:.3f}, {max_xyz[0]:.3f}], 尺寸: {size_xyz[0]:.3f}")
        print(f"  Y范围: [{min_xyz[1]:.3f}, {max_xyz[1]:.3f}], 尺寸: {size_xyz[1]:.3f}")
        print(f"  Z范围: [{min_xyz[2]:.3f}, {max_xyz[2]:.3f}], 尺寸: {size_xyz[2]:.3f}")
        print(
            f"  在PC_MIN-PC_MAX区间内的点: {in_bounds_count}/{len(world_points)} ({in_bounds_ratio * 100:.1f}%)"
        )

    # 创建可视化对象列表
    plotly_objects = []

    # 根据vis_mode决定显示哪种点云
    if vis_mode == "obj":
        # 添加电钻点云
        plotly_objects.extend(Vis.pc(world_points, color="red", size=2))
    elif vis_mode == "all" and obs is not None:
        # 显示所有点云（采样以减少文件大小）
        full_pc_camera, full_pc_world = get_pc_from_obs(obs)

        # 对完整点云进行采样以减少可视化文件大小
        if len(full_pc_world) > 20000:  # 如果点数超过20k，进行采样
            sample_indices = np.random.choice(
                len(full_pc_world), size=20000, replace=False
            )
            full_pc_world_sampled = full_pc_world[sample_indices]
        else:
            full_pc_world_sampled = full_pc_world

        plotly_objects.extend(Vis.pc(full_pc_world_sampled, color="gray", size=1))
        # 同时显示电钻点云（用不同颜色突出显示）
        plotly_objects.extend(Vis.pc(world_points, color="red", size=2))

        if verbose:
            print(
                f"  -> 完整点云: {len(full_pc_world)} 个点，可视化采样: {len(full_pc_world_sampled)} 个点"
            )
    else:
        # 默认显示电钻点云
        plotly_objects.extend(Vis.pc(world_points, color="red", size=2))

    # 添加计算出的边界框
    plotly_objects.extend(
        Vis.box(scale=size_xyz, trans=center_xyz, color="blue", opacity=0.3)
    )

    # 添加相机位姿
    camera_trans = camera_pose[:3, 3]
    camera_rot = camera_pose[:3, :3]
    plotly_objects.extend(Vis.pose(camera_trans, camera_rot, length=0.1))

    # 添加物体真实位姿
    object_trans = object_pose[:3, 3]
    object_rot = object_pose[:3, :3]
    plotly_objects.extend(Vis.pose(object_trans, object_rot, length=0.15))

    # 添加 constants.py 中定义的裁剪区域方框
    constants_center = (PC_MIN + PC_MAX) / 2
    constants_size = PC_MAX - PC_MIN
    plotly_objects.extend(
        Vis.box(
            scale=constants_size, trans=constants_center, color="yellow", opacity=0.2
        )
    )

    # 绘制世界坐标系 xyz 轴 - 使用射线形式
    # 计算场景范围
    all_scene_points = [
        world_points,
        camera_trans.reshape(1, -1),
        object_trans.reshape(1, -1),
        constants_center.reshape(1, -1),
    ]

    scene_points = np.vstack(all_scene_points)

    min_coords = np.min(scene_points, axis=0) - 0.3
    max_coords = np.max(scene_points, axis=0) + 0.3

    # 绘制从原点到场景范围的射线形式的坐标轴
    world_origin = np.array([0.0, 0.0, 0.0])

    # X轴 (红色)
    x_end = np.array([max_coords[0], 0.0, 0.0])
    plotly_objects.extend(Vis.line(world_origin, x_end, color="red", width=4))

    # Y轴 (绿色)
    y_end = np.array([0.0, max_coords[1], 0.0])
    plotly_objects.extend(Vis.line(world_origin, y_end, color="green", width=4))

    # Z轴 (蓝色)
    z_end = np.array([0.0, 0.0, max_coords[2]])
    plotly_objects.extend(Vis.line(world_origin, z_end, color="blue", width=4))

    # 保存可视化到HTML文件
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, f"{sample_id}_drill_bbox.html")
    Vis.show(plotly_objects, path=html_path)

    if verbose:
        print(f"  可视化已保存到: {html_path}")
        print(f"  constants裁剪区域:")
        print(
            f"    x范围：[{constants_center[0] - constants_size[0] / 2:.3f}, {constants_center[0] + constants_size[0] / 2:.3f}]"
        )
        print(
            f"    y范围：[{constants_center[1] - constants_size[1] / 2:.3f}, {constants_center[1] + constants_size[1] / 2:.3f}]"
        )
        print(
            f"    z范围：[{constants_center[2] - constants_size[2] / 2:.3f}, {constants_center[2] + constants_size[2] / 2:.3f}]"
        )

    return html_path


def visualize_all_driller_bbox(
    all_vis_data, output_dir="vis_output", verbose=False, vis_mode="obj"
):
    """可视化所有样本的电钻点云及其边界框"""
    if len(all_vis_data) == 0:
        if verbose:
            print("没有找到任何可视化数据")
        return None

    if verbose:
        print(f"开始生成 {len(all_vis_data)} 个样本的综合可视化...")

    # 创建可视化对象列表
    plotly_objects = []

    # 为每个样本分配不同的颜色
    import colorsys

    colors = []
    for i in range(len(all_vis_data)):
        hue = i / len(all_vis_data)
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        color = f"rgb({int(rgb[0] * 255)}, {int(rgb[1] * 255)}, {int(rgb[2] * 255)})"
        colors.append(color)

    # 收集所有点云和位姿信息用于确定场景范围
    all_scene_points = []

    # 处理每个样本
    for idx, vis_data in enumerate(all_vis_data):
        driller_points = vis_data["driller_points"]
        camera_pose = vis_data["camera_pose"]
        object_pose = vis_data["object_pose"]
        sample_id = vis_data["sample_id"]

        if len(driller_points) == 0:
            continue

        # 将点云从相机坐标系转换到世界坐标系
        points_homo = np.hstack([driller_points, np.ones((len(driller_points), 1))])
        world_points = (camera_pose @ points_homo.T).T[:, :3]

        # 收集场景点
        all_scene_points.append(world_points)
        all_scene_points.append(camera_pose[:3, 3].reshape(1, -1))
        all_scene_points.append(object_pose[:3, 3].reshape(1, -1))

        # 计算边界框
        min_xyz = np.min(world_points, axis=0)
        max_xyz = np.max(world_points, axis=0)
        center_xyz = (min_xyz + max_xyz) / 2
        size_xyz = max_xyz - min_xyz

        # 根据vis_mode决定显示哪种点云
        if vis_mode == "obj":
            # 只显示电钻点云
            plotly_objects.extend(Vis.pc(world_points, color=colors[idx], size=2))
        elif vis_mode == "all" and "obs" in vis_data:
            # 显示所有点云（采样以减少文件大小）
            obs = vis_data["obs"]
            full_pc_camera, full_pc_world = get_pc_from_obs(obs)

            # 对完整点云进行采样以减少可视化文件大小
            if len(full_pc_world) > 10000:  # 综合可视化中使用更少的点
                sample_indices = np.random.choice(
                    len(full_pc_world), size=10000, replace=False
                )
                full_pc_world_sampled = full_pc_world[sample_indices]
            else:
                full_pc_world_sampled = full_pc_world

            plotly_objects.extend(
                Vis.pc(full_pc_world_sampled, color="lightgray", size=1)
            )
            # 同时显示电钻点云（用不同颜色突出显示）
            plotly_objects.extend(Vis.pc(world_points, color=colors[idx], size=2))
        else:
            # 默认显示电钻点云
            plotly_objects.extend(Vis.pc(world_points, color=colors[idx], size=2))

        # 添加计算出的边界框
        plotly_objects.extend(
            Vis.box(scale=size_xyz, trans=center_xyz, color=colors[idx], opacity=0.1)
        )

        # 添加相机位姿 (较小的尺寸以避免过于杂乱)
        # camera_trans = camera_pose[:3, 3]
        # camera_rot = camera_pose[:3, :3]
        # plotly_objects.extend(Vis.pose(camera_trans, camera_rot, length=0.05))

        # # 添加物体真实位姿 (较小的尺寸)
        # object_trans = object_pose[:3, 3]
        # object_rot = object_pose[:3, :3]
        # plotly_objects.extend(Vis.pose(object_trans, object_rot, length=0.08))

    # 添加 constants.py 中定义的裁剪区域方框
    constants_center = (PC_MIN + PC_MAX) / 2
    constants_size = PC_MAX - PC_MIN
    plotly_objects.extend(
        Vis.box(
            scale=constants_size, trans=constants_center, color="yellow", opacity=0.15
        )
    )
    all_scene_points.append(constants_center.reshape(1, -1))

    # 计算场景范围
    if all_scene_points:
        scene_points = np.vstack(all_scene_points)
        min_coords = np.min(scene_points, axis=0) - 0.3
        max_coords = np.max(scene_points, axis=0) + 0.3

        # 绘制世界坐标系轴
        world_origin = np.array([0.0, 0.0, 0.0])

        # X轴 (红色)
        x_end = np.array([max_coords[0], 0.0, 0.0])
        plotly_objects.extend(Vis.line(world_origin, x_end, color="red", width=6))

        # Y轴 (绿色)
        y_end = np.array([0.0, max_coords[1], 0.0])
        plotly_objects.extend(Vis.line(world_origin, y_end, color="green", width=6))

        # Z轴 (蓝色)
        z_end = np.array([0.0, 0.0, max_coords[2]])
        plotly_objects.extend(Vis.line(world_origin, z_end, color="blue", width=6))

    # 保存可视化到HTML文件
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, "all.html")
    Vis.show(plotly_objects, path=html_path)

    if verbose:
        print(f"综合可视化包含 {len(all_vis_data)} 个样本")
        print(f"已保存到: {html_path}")

    return html_path


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()

    # 确定是否使用详细模式
    verbose = args.verbose

    if verbose:
        print("=" * 50)
        print("电钻位姿数据集生成器")
        print("=" * 50)
        if args.range is not None:
            start_idx, end_idx = map(int, args.range.split(","))
            print(
                f"样本范围: [{start_idx}, {end_idx}) (共 {end_idx - start_idx} 个样本)"
            )
        else:
            print(f"样本数量: {args.num_samples}")
        print(f"输出目录: {args.output_dir}")
        print(f"物体名称: {args.object_name}")
        print(f"训练比例: {args.train_ratio}")
        print(f"可视化模式: {args.vis}")
        if args.vis:
            print(f"单独可视化: {args.individual}")
            print(f"可视化点云类型: {args.vis_mode}")
        print(f"保存点云: {args.save_pc}")
        print("=" * 50)

    # 生成数据
    train_dir, val_dir = generate_data(args)

    # 测试生成的数据
    test_generated_data(train_dir, val_dir, verbose=verbose)
