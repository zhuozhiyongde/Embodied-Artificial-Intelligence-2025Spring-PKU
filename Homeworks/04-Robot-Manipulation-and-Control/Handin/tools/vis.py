import numpy as np
import plotly.graph_objects as go
import plotly.offline as offline
import os
import argparse
import random
import re
import glob
from src.constants import PC_MIN, PC_MAX

# 在这里定义要可视化的文件正则表达式列表
# 用户可以修改这个列表来指定要可视化的文件模式
# 系统会自动搜索匹配的文件，识别文件类型（点云或位姿）并选择合适的可视化方式
VISUALIZATION_FILES = [
    # 示例用法（正则表达式模式）：
    # r"sample_data/.*\.npy$",                                    # 匹配sample_data目录下所有.npy文件
    # r".*/full_pc_world.*\.npy$",                               # 匹配任何路径下包含full_pc_world的.npy文件
    # r".*/object_pose\.npy$",                                   # 匹配任何路径下的object_pose.npy文件
    # ------------------------------
    # r"./model_sample_data/.*\.npy$",
    r"./main_sample_data/.*\.npy$",
    # r"./main_sample_data/pc_camera.npy$",
    # ------------------------------
    # r"./data/power_drill/train/sample_000000/full_pc_world_raw.npy$",
    # r"./data/power_drill/(train|val)/sample_000000/driller_pc_world.npy$",
    # ------------------------------
    # r"./test_data/power_drill/(train|val)/sample_000000/full_pc_world_raw.npy$",
    # r"./test_data/power_drill/(train|val)/sample_\d+/driller_pc_world.npy$",
]

EXCLUDE_REGEX = r".*/camera_pose.npy"

# 默认颜色池（用于随机选择）
DEFAULT_COLORS = [
    # "red",
    # "blue",
    # "green",
    "orange",
    "purple",
    "cyan",
    "magenta",
    "yellow",
    "pink",
    "brown",
    "lightblue",
    "lightgreen",
    "coral",
    "gold",
    "indigo",
    "lime",
]


def calculate_table_height(pc, z_min=0.68, z_max=0.78):
    """计算桌面高度"""
    # 筛选出 z 值在范围内的点
    filtered_points = pc[
        (pc[:, 0] > PC_MIN[0])
        & (pc[:, 0] < PC_MAX[0])
        & (pc[:, 1] > PC_MIN[1])
        & (pc[:, 1] < PC_MAX[1])
        & (pc[:, 2] >= z_min)
        & (pc[:, 2] <= z_max)
    ]
    if len(filtered_points) == 0:
        raise ValueError("筛选后没有点，检查输入或调整 z 值范围")

    # 计算 z 均值
    table_height = np.mean(filtered_points[:, 2])
    return table_height


def get_workspace_mask(pc: np.ndarray) -> np.ndarray:
    """获取工作空间内点云的掩码"""
    table_height = calculate_table_height(pc)
    pc_mask = (
        (pc[:, 0] > PC_MIN[0])
        & (pc[:, 0] < PC_MAX[0])
        & (pc[:, 1] > PC_MIN[1])
        & (pc[:, 1] < PC_MAX[1])
        & (pc[:, 2] > table_height + 0.005)
        & (pc[:, 2] < PC_MAX[2])
    )
    return pc_mask


def create_box_edges(min_point, max_point):
    """创建边界框的边线"""
    # 定义8个顶点
    vertices = np.array(
        [
            [min_point[0], min_point[1], min_point[2]],
            [max_point[0], min_point[1], min_point[2]],
            [max_point[0], max_point[1], min_point[2]],
            [min_point[0], max_point[1], min_point[2]],
            [min_point[0], min_point[1], max_point[2]],
            [max_point[0], min_point[1], max_point[2]],
            [max_point[0], max_point[1], max_point[2]],
            [min_point[0], max_point[1], max_point[2]],
        ]
    )

    # 定义边的连接关系
    edges = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],  # 底面
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],  # 顶面
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],  # 垂直边
    ]

    return vertices, edges


def add_bounding_box(fig, min_point, max_point, color="red", name="边界框"):
    """向图形添加边界框"""
    vertices, edges = create_box_edges(min_point, max_point)

    # 添加边线
    for i, edge in enumerate(edges):
        showlegend = i == 0  # 只在第一条边显示图例
        fig.add_trace(
            go.Scatter3d(
                x=[vertices[edge[0]][0], vertices[edge[1]][0]],
                y=[vertices[edge[0]][1], vertices[edge[1]][1]],
                z=[vertices[edge[0]][2], vertices[edge[1]][2]],
                mode="lines",
                line=dict(color=color, width=6),
                name=name if showlegend else "",
                showlegend=showlegend,
                hoverinfo="skip",
            )
        )

    # 添加角点
    fig.add_trace(
        go.Scatter3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            mode="markers",
            marker=dict(
                size=8,
                color=color,
                symbol="diamond",
                line=dict(color="darkred" if color == "red" else "black", width=2),
            ),
            name=f"{name}角点",
            showlegend=False,
            hoverinfo="skip",
        )
    )


def find_files_by_patterns(patterns, verbose=True):
    """根据正则表达式模式查找匹配的文件"""
    matched_files = []

    for pattern in patterns:
        if verbose:
            print(f"🔍 搜索模式: {pattern}")

        # 编译正则表达式
        try:
            regex = re.compile(pattern)
        except re.error as e:
            print(f"❌ 无效的正则表达式 '{pattern}': {e}")
            continue

        pattern_matches = []

        # 从当前目录开始递归搜索所有.npy文件，然后用正则表达式过滤
        # 搜索从当前目录开始，因为正则表达式包含完整路径
        search_pattern = "./**/*.npy"
        found_files = glob.glob(search_pattern, recursive=True)

        # 用正则表达式过滤
        for file_path in found_files:
            if regex.search(file_path) and not re.search(EXCLUDE_REGEX, file_path):
                pattern_matches.append(os.path.abspath(file_path))

                # 去除重复但保持顺序
        seen = set()
        unique_pattern_matches = []
        for match in pattern_matches:
            if match not in seen:
                seen.add(match)
                unique_pattern_matches.append(match)

        matched_files.extend(unique_pattern_matches)

    # 去除重复文件但保持顺序
    seen = set()
    unique_files = []
    for file in matched_files:
        if file not in seen:
            seen.add(file)
            unique_files.append(file)

    if verbose:
        print(f"\n📁 总共找到 {len(unique_files)} 个唯一文件")

    return unique_files


def visualize_multiple_pointclouds(
    output_path="multi_pointcloud_visualization.html",
):
    """可视化多个点云和位姿文件"""

    # 根据正则表达式模式查找文件
    if not VISUALIZATION_FILES:
        print("❌ 错误: 没有指定文件模式!")
        print("请修改脚本顶部的 VISUALIZATION_FILES 列表，添加要可视化的正则表达式模式")
        return

    print("🔍 开始根据正则表达式模式搜索文件...")
    file_list = find_files_by_patterns(VISUALIZATION_FILES, verbose=True)

    if not file_list:
        print("❌ 错误: 没有找到匹配的文件!")
        print("请检查正则表达式模式是否正确，以及文件是否存在")
        return

    print(f"开始可视化 {len(file_list)} 个文件...")

    # 创建图形
    fig = go.Figure()

    all_points = []
    valid_items = 0
    table_height = None
    valid_points = 0
    used_colors = set()
    pose_count = 0
    pointcloud_count = 0

    for i, file_path in enumerate(file_list):
        try:
            # 加载数据
            if not os.path.exists(file_path):
                print(f"⚠️  文件不存在: {file_path}")
                continue

            data = np.load(file_path)
            display_name = get_display_name(file_path)

            if len(data) == 0:
                print(f"⚠️  空文件: {file_path}")
                continue

            # 判断文件类型并处理
            if is_pose_file(data):
                # 处理位姿数据
                print(f"✅ 加载位姿 {pose_count + 1}: {file_path}")
                print(f"   形状: {data.shape}")

                if data.shape == (4, 4):
                    print(f"   位姿矩阵: {data}")
                    position, rotation = extract_pose_from_matrix(data)
                    # 验证 rotation 是否是正交矩阵
                    print(f"   旋转矩阵: {rotation}")
                    print(
                        f"   旋转矩阵是否是正交矩阵: {np.allclose(rotation @ rotation.T, np.eye(3))}"
                    )
                    print(
                        f"   位置: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]"
                    )

                    # 计算合适的坐标轴长度（基于数据范围）
                    scale = 0.30

                    print(f"   坐标轴长度: {scale:.3f}")

                    # 添加坐标系到图形
                    add_coordinate_frame(
                        fig,
                        position,
                        rotation,
                        scale=scale,
                        name=display_name,
                        show_legend=True,
                    )

                    # 将位置添加到all_points用于计算范围
                    all_points.append(position.reshape(1, -1))
                    pose_count += 1
                else:
                    print(f"⚠️  暂不支持的位姿格式: {data.shape}")
                    continue

            else:
                # 处理点云数据
                if len(data.shape) != 2 or data.shape[1] != 3:
                    print(f"⚠️  无效的点云格式: {data.shape}")
                    continue

                # 如果这是第一个有效的点云，计算桌面高度
                if table_height is None:
                    try:
                        table_height = calculate_table_height(data)
                        valid_points = np.sum(get_workspace_mask(data))
                        print(f"📏 计算出的桌面高度: {table_height:.3f}")
                    except ValueError as e:
                        print(f"⚠️  无法计算桌面高度: {e}")
                        table_height = 0.72  # 使用默认值

                print(f"✅ 加载点云 {pointcloud_count + 1}: {file_path}")
                print(f"   点数: {len(data)}")
                print(
                    f"   范围: X[{np.min(data[:, 0]):.3f}, {np.max(data[:, 0]):.3f}], "
                    f"Y[{np.min(data[:, 1]):.3f}, {np.max(data[:, 1]):.3f}], "
                    f"Z[{np.min(data[:, 2]):.3f}, {np.max(data[:, 2]):.3f}]"
                )

                # 为了更好地可视化，对大点云进行采样
                n_sample = min(50000, len(data))
                if len(data) > n_sample:
                    indices = np.random.choice(len(data), n_sample, replace=False)
                    pc_sample = data[indices]
                    print(f"   采样到 {n_sample} 个点用于显示")
                else:
                    pc_sample = data

                # 选择随机颜色
                color = select_random_color(used_colors, DEFAULT_COLORS)

                # 添加点云到图形
                fig.add_trace(
                    go.Scatter3d(
                        x=pc_sample[:, 0],
                        y=pc_sample[:, 1],
                        z=pc_sample[:, 2],
                        mode="markers",
                        marker=dict(
                            size=3,
                            color=color,
                            opacity=0.7,
                        ),
                        name=display_name,
                        hovertemplate=f"<b>{display_name}</b><br>"
                        "X: %{x:.3f}<br>"
                        "Y: %{y:.3f}<br>"
                        "Z: %{z:.3f}<br>"
                        "<extra></extra>",
                    )
                )

                all_points.append(data)
                pointcloud_count += 1

            valid_items += 1

        except Exception as e:
            print(f"❌ 加载 {file_path} 时出错: {e}")
            continue

    if valid_items == 0:
        print("❌ 没有成功加载任何数据!")
        return

    # 处理边界框和范围计算
    if all_points:
        # 合并所有点云数据用于计算范围
        combined_points = np.vstack(all_points)

        # 如果有点云数据且计算了桌面高度，添加相关边界框
        if table_height is not None:
            # 创建动态工作空间边界 (使用计算出的桌面高度 + 0.005 作为下边界)
            dynamic_pc_min = np.array([PC_MIN[0], PC_MIN[1], table_height + 0.005])
            dynamic_pc_max = PC_MAX

            # 添加原始常量定义的边界框
            add_bounding_box(
                fig, PC_MIN, PC_MAX, color="yellow", name="原始裁剪区域 (PC_MIN/PC_MAX)"
            )

            # 添加动态计算的工作空间边界框
            add_bounding_box(
                fig,
                dynamic_pc_min,
                dynamic_pc_max,
                color="orange",
                name="动态工作空间边界",
            )

            # 添加桌面高度参考平面
            fig.add_trace(
                go.Scatter3d(
                    x=[PC_MIN[0], PC_MAX[0], PC_MAX[0], PC_MIN[0], PC_MIN[0]],
                    y=[PC_MIN[1], PC_MIN[1], PC_MAX[1], PC_MAX[1], PC_MIN[1]],
                    z=[
                        table_height,
                        table_height,
                        table_height,
                        table_height,
                        table_height,
                    ],
                    mode="lines",
                    line=dict(color="red", width=4),
                    name="计算出的桌面高度",
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

        # 添加实际数据范围的边界框
        data_min = np.min(combined_points, axis=0)
        data_max = np.max(combined_points, axis=0)
        add_bounding_box(fig, data_min, data_max, color="gray", name="数据实际范围")

        # 计算合适的视图范围
        if table_height is not None:
            dynamic_pc_min = np.array([PC_MIN[0], PC_MIN[1], table_height + 0.005])
            all_scene_points = np.vstack(
                [combined_points, dynamic_pc_min.reshape(1, -1), PC_MAX.reshape(1, -1)]
            )
        else:
            all_scene_points = combined_points

        x_range = [np.min(all_scene_points[:, 0]), np.max(all_scene_points[:, 0])]
        y_range = [np.min(all_scene_points[:, 1]), np.max(all_scene_points[:, 1])]
        z_range = [np.min(all_scene_points[:, 2]), np.max(all_scene_points[:, 2])]
    else:
        # 如果没有点云数据，使用默认范围
        x_range = [-1, 1]
        y_range = [-1, 1]
        z_range = [0, 2]
        data_min = np.array([-1, -1, 0])
        data_max = np.array([1, 1, 2])

    # 扩展范围以便更好地观察
    x_padding = (x_range[1] - x_range[0]) * 0.1
    y_padding = (y_range[1] - y_range[0]) * 0.1
    z_padding = (z_range[1] - z_range[0]) * 0.1

    # 确保有最小的范围和padding，特别是对于只有位姿数据的情况
    min_range = 0.2  # 最小范围20cm
    min_padding = 0.1  # 最小padding 10cm

    if (x_range[1] - x_range[0]) < min_range:
        center_x = (x_range[0] + x_range[1]) / 2
        x_range = [center_x - min_range / 2, center_x + min_range / 2]
        x_padding = max(x_padding, min_padding)

    if (y_range[1] - y_range[0]) < min_range:
        center_y = (y_range[0] + y_range[1]) / 2
        y_range = [center_y - min_range / 2, center_y + min_range / 2]
        y_padding = max(y_padding, min_padding)

    if (z_range[1] - z_range[0]) < min_range:
        center_z = (z_range[0] + z_range[1]) / 2
        z_range = [center_z - min_range / 2, center_z + min_range / 2]
        z_padding = max(z_padding, min_padding)

    # 设置布局
    title_text = f"数据可视化 - {pointcloud_count} 个点云, {pose_count} 个位姿<br>"
    if table_height is not None:
        title_text += f"<sub>🟡原始裁剪区域, 🟠动态工作空间边界(桌面高度+0.005={table_height + 0.005:.3f}), 🔴桌面高度({table_height:.3f}), 🔘数据实际范围</sub>"
    else:
        title_text += f"<sub>🔘数据实际范围</sub>"

    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
        ),
        scene=dict(
            xaxis_title="X 坐标",
            yaxis_title="Y 坐标",
            zaxis_title="Z 坐标",
            aspectmode="data",  # 使用data模式，让Plotly根据数据自动保持比例
            xaxis=dict(
                range=[x_range[0] - x_padding, x_range[1] + x_padding],
                showgrid=True,
                gridwidth=1,
                gridcolor="lightgray",
            ),
            yaxis=dict(
                range=[y_range[0] - y_padding, y_range[1] + y_padding],
                showgrid=True,
                gridwidth=1,
                gridcolor="lightgray",
            ),
            zaxis=dict(
                range=[z_range[0] - z_padding, z_range[1] + z_padding],
                showgrid=True,
                gridwidth=1,
                gridcolor="lightgray",
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
            ),
        ),
        width=1400,
        height=1000,
        legend=dict(
            x=0,
            y=1,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1,
        ),
    )

    # 保存为HTML文件
    offline.plot(fig, filename=output_path, auto_open=False)

    print(f"\n✅ 可视化文件已保存至: {output_path}")
    print(f"📊 统计信息:")
    print(f"  - 成功加载的点云数量: {pointcloud_count}")
    print(f"  - 成功加载的位姿数量: {pose_count}")
    if all_points and len(combined_points) > 0:
        print(f"  - 总点数: {len(combined_points):,}")
    if table_height is not None:
        print(f"  - 计算出的桌面高度: {table_height:.3f}")
        print(f"  - 有效点数: {valid_points}")
    print(
        f"  - 数据范围: X[{data_min[0]:.3f}, {data_max[0]:.3f}], Y[{data_min[1]:.3f}, {data_max[1]:.3f}], Z[{data_min[2]:.3f}, {data_max[2]:.3f}]"
    )
    if table_height is not None:
        print(
            f"  - 原始裁剪区域: X[{PC_MIN[0]:.3f}, {PC_MAX[0]:.3f}], Y[{PC_MIN[1]:.3f}, {PC_MAX[1]:.3f}], Z[{PC_MIN[2]:.3f}, {PC_MAX[2]:.3f}]"
        )
        dynamic_pc_min_final = np.array([PC_MIN[0], PC_MIN[1], table_height + 0.005])
        print(
            f"  - 动态工作空间边界: X[{dynamic_pc_min_final[0]:.3f}, {PC_MAX[0]:.3f}], Y[{dynamic_pc_min_final[1]:.3f}, {PC_MAX[1]:.3f}], Z[{dynamic_pc_min_final[2]:.3f}, {PC_MAX[2]:.3f}]"
        )

    return fig


def parse_args():
    parser = argparse.ArgumentParser(description="可视化多个点云文件")
    parser.add_argument(
        "--output",
        type=str,
        default="visualization.html",
        help="输出HTML文件路径",
    )
    return parser.parse_args()


def is_pose_file(data: np.ndarray) -> bool:
    """判断数据是否为位姿数据"""
    # 检查是否为 4x4 变换矩阵
    if data.shape == (4, 4):
        # 检查最后一行是否为 [0, 0, 0, 1]
        if np.allclose(data[3, :], [0, 0, 0, 1]):
            return True
    # 检查是否为 7维位姿 (x, y, z, qx, qy, qz, qw)
    elif data.shape == (7,):
        return True
    # 检查是否为 6维位姿 (x, y, z, rx, ry, rz)
    elif data.shape == (6,):
        return True
    return False


def extract_pose_from_matrix(pose_matrix: np.ndarray) -> tuple:
    """从4x4变换矩阵中提取位置和旋转矩阵"""
    if pose_matrix.shape != (4, 4):
        raise ValueError("位姿矩阵必须是4x4")

    position = pose_matrix[:3, 3]
    rotation = pose_matrix[:3, :3]
    return position, rotation


def add_coordinate_frame(
    fig,
    position: np.ndarray,
    rotation: np.ndarray,
    scale: float = 0.1,
    name: str = "坐标系",
    show_legend: bool = True,
):
    """向图形添加坐标系"""
    # 定义坐标轴方向
    axes = np.array(
        [
            [scale, 0, 0],  # X轴 (红色)
            [0, scale, 0],  # Y轴 (绿色)
            [0, 0, scale],  # Z轴 (蓝色)
        ]
    )

    # 应用旋转
    axes_rotated = rotation @ axes.T

    # 计算坐标轴终点
    axis_ends = position.reshape(-1, 1) + axes_rotated

    colors = ["red", "green", "blue"]
    axis_names = ["X", "Y", "Z"]

    for i, (color, axis_name) in enumerate(zip(colors, axis_names)):
        showlegend = (i == 0) and show_legend
        legend_name = f"{name}" if showlegend else ""

        # 添加轴线
        fig.add_trace(
            go.Scatter3d(
                x=[position[0], axis_ends[0, i]],
                y=[position[1], axis_ends[1, i]],
                z=[position[2], axis_ends[2, i]],
                mode="lines",
                line=dict(color=color, width=8),
                name=legend_name,
                showlegend=showlegend,
                hovertemplate=f"<b>{name} - {axis_name}轴</b><br>"
                + "起点: (%{x:.3f}, %{y:.3f}, %{z:.3f})<br>"
                + "<extra></extra>",
            )
        )

        # 添加箭头头部
        fig.add_trace(
            go.Scatter3d(
                x=[axis_ends[0, i]],
                y=[axis_ends[1, i]],
                z=[axis_ends[2, i]],
                mode="markers",
                marker=dict(
                    size=12,
                    color=color,
                    symbol="diamond",
                    line=dict(color="black", width=2),
                ),
                showlegend=False,
                hovertemplate=f"<b>{name} - {axis_name}轴终点</b><br>"
                + "位置: (%{x:.3f}, %{y:.3f}, %{z:.3f})<br>"
                + "<extra></extra>",
            )
        )

    # 添加原点
    fig.add_trace(
        go.Scatter3d(
            x=[position[0]],
            y=[position[1]],
            z=[position[2]],
            mode="markers",
            marker=dict(
                size=15,
                color="black",
                symbol="circle",
                line=dict(color="white", width=3),
            ),
            name=f"{name}原点" if show_legend else "",
            showlegend=show_legend,
            hovertemplate=f"<b>{name}原点</b><br>"
            + "位置: (%{x:.3f}, %{y:.3f}, %{z:.3f})<br>"
            + "<extra></extra>",
        )
    )


def get_display_name(file_path: str) -> str:
    """从文件路径提取显示名称"""
    filename = os.path.basename(file_path)
    name_without_ext = os.path.splitext(filename)[0]

    # 将下划线替换为空格，并进行简单的名称美化
    display_name = name_without_ext.replace("_", " ")

    # 一些常见的名称替换
    replacements = {
        "full pc world": "完整场景点云",
        "driller pc world": "电钻点云",
        "object pose": "物体位姿",
        "world": "世界坐标系",
        "raw": "原始数据",
    }

    for old, new in replacements.items():
        if old in display_name.lower():
            display_name = display_name.lower().replace(old, new)
            break

    return display_name.title()


def select_random_color(used_colors: set, available_colors: list) -> str:
    """选择一个未使用的随机颜色"""
    available = [c for c in available_colors if c not in used_colors]
    if not available:
        # 如果所有颜色都用完了，重新开始使用
        available = available_colors
        used_colors.clear()

    color = available[0]
    used_colors.add(color)
    return color


if __name__ == "__main__":
    args = parse_args()

    # 使用脚本顶部定义的全局变量
    if not VISUALIZATION_FILES:
        print("=" * 60)
        print("📝 使用说明:")
        print("请修改脚本顶部的 VISUALIZATION_FILES 列表，添加正则表达式模式")
        print("=" * 60)
        print("\n📄 示例配置:")
        print("VISUALIZATION_FILES = [")
        print(
            '    r".*/full_pc_world.*\\.npy$",  # 匹配任何路径下包含full_pc_world的.npy文件'
        )
        print(
            '    r".*/driller_pc_world\\.npy$",  # 匹配任何路径下的driller_pc_world.npy文件'
        )
        print(
            '    r".*/object_pose\\.npy$",       # 匹配任何路径下的object_pose.npy文件'
        )
        print(
            '    r"data/power_drill/train/sample_00000[0-4]/.*\\.npy$",  # 匹配指定样本范围的所有.npy文件'
        )
        print("]")
        print("\n✨ 新功能:")
        print("- 支持正则表达式模式自动搜索文件")
        print("- 支持点云和位姿文件混合可视化")
        print("- 自动从文件名提取显示名称")
        print("- 随机颜色分配")
        print("- 支持任意数量的文件")
    else:
        visualize_multiple_pointclouds(output_path=args.output)
