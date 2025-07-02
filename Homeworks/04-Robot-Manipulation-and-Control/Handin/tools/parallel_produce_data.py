#!/usr/bin/env python3
"""
并行数据生成脚本
使用多进程并行生成电钻位姿数据集
总共生成5000个样本，分为3个进程并行处理
"""

import argparse
import multiprocessing as mp
import subprocess
import sys
import time
import os
from pathlib import Path


def run_data_generation(
    num_samples,
    range_str,
    output_dir,
    object_name,
    train_ratio,
    verbose,
    save_pc,
    vis,
    individual,
    vis_mode,
):
    """运行单个数据生成进程"""
    process_id = mp.current_process().name
    print(f"[{process_id}] 开始生成范围 {range_str} 的数据...")

    # 构建命令行参数
    cmd = [
        sys.executable,
        "produce_data.py",
        "--num_samples",
        str(num_samples),
        "--range",
        range_str,
        "--output_dir",
        output_dir,
        "--object_name",
        object_name,
        "--train_ratio",
        str(train_ratio),
    ]

    if verbose:
        cmd.append("--verbose")

    if save_pc:
        cmd.append("--save_pc")

    if vis:
        cmd.append("--vis")

    if individual:
        cmd.append("--individual")

    if vis_mode:
        cmd.extend(["--vis_mode", vis_mode])

    try:
        # 运行子进程
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        end_time = time.time()

        elapsed_time = end_time - start_time
        print(f"[{process_id}] 范围 {range_str} 生成完成，耗时: {elapsed_time:.2f} 秒")

        if verbose:
            print(f"[{process_id}] stdout:")
            print(result.stdout)

        return True, range_str, elapsed_time

    except subprocess.CalledProcessError as e:
        print(f"[{process_id}] 范围 {range_str} 生成失败:")
        print(f"错误代码: {e.returncode}")
        print(f"stderr: {e.stderr}")
        if verbose:
            print(f"stdout: {e.stdout}")
        return False, range_str, 0
    except Exception as e:
        print(f"[{process_id}] 范围 {range_str} 生成时发生异常: {e}")
        return False, range_str, 0


def parse_args():
    parser = argparse.ArgumentParser(description="并行生成电钻位姿数据集")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="总样本数量 (默认: 5000)",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=4,
        help="并行进程数 (默认: 4)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="输出目录 (默认: data)",
    )
    parser.add_argument(
        "--object_name",
        type=str,
        default="power_drill",
        help="物体名称 (默认: power_drill)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="训练集比例 (默认: 0.8)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细模式，启用所有打印输出",
    )
    parser.add_argument(
        "--save_pc",
        action="store_true",
        default=False,
        help="保存点云数据",
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
        "--vis_mode",
        type=str,
        choices=["obj", "all"],
        default="obj",
        help="可视化模式：obj=只显示物体点云，all=显示所有点云 (默认: obj)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    total_samples = args.num_samples
    num_processes = args.num_processes

    print("=" * 60)
    print("并行电钻位姿数据集生成器")
    print("=" * 60)
    print(f"总样本数: {total_samples}")
    print(f"并行进程数: {num_processes}")
    print(f"输出目录: {args.output_dir}")
    print(f"物体名称: {args.object_name}")
    print(f"训练比例: {args.train_ratio}")
    print(f"详细模式: {args.verbose}")
    print(f"保存点云: {args.save_pc}")
    if args.vis:
        print(f"可视化模式: {args.vis}")
        print(f"单独可视化: {args.individual}")
        print(f"可视化点云类型: {args.vis_mode}")
    print("=" * 60)

    # 检查 produce_data.py 是否存在
    if not Path("produce_data.py").exists():
        print("错误: 找不到 produce_data.py 文件")
        sys.exit(1)

    # 计算每个进程的样本范围
    samples_per_process = total_samples // num_processes
    remaining_samples = total_samples % num_processes

    ranges = []
    start_idx = 0

    for i in range(num_processes):
        # 将剩余样本分配给前几个进程
        current_samples = samples_per_process
        if i < remaining_samples:
            current_samples += 1

        end_idx = start_idx + current_samples
        ranges.append(f"{start_idx},{end_idx}")

        print(
            f"进程 {i + 1}: 样本范围 [{start_idx}, {end_idx}) - {current_samples} 个样本"
        )
        start_idx = end_idx

    print("=" * 60)
    print("开始并行生成数据...")

    # 创建进程池并执行
    start_time = time.time()

    with mp.Pool(processes=num_processes) as pool:
        # 为每个进程准备参数
        tasks = []
        for i, range_str in enumerate(ranges):
            task_args = (
                total_samples,
                range_str,
                args.output_dir,
                args.object_name,
                args.train_ratio,
                args.verbose,
                args.save_pc,
                args.vis,
                args.individual,
                args.vis_mode,
            )
            tasks.append(task_args)

        # 使用starmap执行任务
        results = pool.starmap(run_data_generation, tasks)

    end_time = time.time()
    total_elapsed = end_time - start_time

    # 统计结果
    successful_ranges = []
    failed_ranges = []
    total_time_spent = 0

    for success, range_str, elapsed in results:
        if success:
            successful_ranges.append(range_str)
            total_time_spent += elapsed
        else:
            failed_ranges.append(range_str)

    print("=" * 60)
    print("并行生成完成!")
    print(f"总耗时: {total_elapsed:.2f} 秒")
    print(f"成功生成的范围: {len(successful_ranges)}/{len(ranges)}")

    if successful_ranges:
        print("成功的范围:")
        for range_str in successful_ranges:
            print(f"  - {range_str}")

    if failed_ranges:
        print("失败的范围:")
        for range_str in failed_ranges:
            print(f"  - {range_str}")
        print("\n可以单独重新运行失败的范围:")
        for range_str in failed_ranges:
            print(
                f"python produce_data.py --range {range_str} --output_dir {args.output_dir} --object_name {args.object_name}"
            )

    # 给出数据集使用建议
    if len(successful_ranges) == len(ranges):
        print(f"\n数据集已成功生成到: {args.output_dir}")
        train_dir = os.path.join(args.output_dir, args.object_name, "train")
        val_dir = os.path.join(args.output_dir, args.object_name, "val")
        print(f"训练集目录: {train_dir}")
        print(f"验证集目录: {val_dir}")
        print("\n可以开始训练模型了!")
    else:
        print(f"\n警告: 只有 {len(successful_ranges)}/{len(ranges)} 个进程成功完成")
        print("请检查失败的范围并重新运行")

    print("=" * 60)


if __name__ == "__main__":
    # 设置多进程启动方法（Linux下通常是fork，但显式设置确保兼容性）
    mp.set_start_method("spawn", force=True)  # 使用spawn方法确保跨平台兼容性
    main()
