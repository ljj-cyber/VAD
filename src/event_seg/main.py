import os
import sys
import pathlib
import argparse
import glob
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading
import time

# 设置路径
_current_dir = pathlib.Path(__file__).parent.resolve()
_src_dir = _current_dir.parent.resolve()
_project_root = _src_dir.parent.resolve()

sys.path.insert(0, str(_current_dir))
sys.path.insert(0, str(_src_dir))
sys.path.insert(0, str(_src_dir / "LAVIS"))


def get_video_files(input_dir):
    """获取所有视频文件"""
    video_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.split('.')[-1].lower() in ['mp4', 'avi', 'mov']:
                video_files.append(os.path.join(root, file))
    return video_files


def get_output_dir(input_path, input_base, output_base):
    """获取视频对应的输出目录"""
    rel_path = os.path.relpath(os.path.dirname(input_path), input_base)
    video_name = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(output_base, rel_path, f"{video_name}_segments")


def is_processed(output_dir, manifest_paths):
    """检查视频是否已处理"""
    if os.path.isdir(output_dir):
        segments = glob.glob(os.path.join(output_dir, "segment_*.mp4"))
        if len(segments) > 0:
            return True
    seg_prefix = os.path.join(output_dir, "segment_")
    return any(p.startswith(seg_prefix) for p in manifest_paths)


def worker_process(gpu_id, video_queue, result_queue, input_base, output_base):
    """
    工作进程：在指定 GPU 上处理视频队列
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    from video_processing import process_video_with_processor
    from uniseg_processor import UniSegProcessor
    
    print(f"[GPU {gpu_id}] 初始化模型...")
    processor = UniSegProcessor()
    print(f"[GPU {gpu_id}] 模型初始化完成")
    
    processed_count = 0
    
    while True:
        try:
            item = video_queue.get(timeout=1)
        except:
            if video_queue.empty():
                break
            continue
        
        if item is None:
            break
        
        input_path = item
        output_dir = get_output_dir(input_path, input_base, output_base)
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            segment_info = process_video_with_processor(processor, input_path, output_dir)
            result_queue.put(('success', input_path, output_dir, segment_info))
            processed_count += 1
            print(f"[GPU {gpu_id}] 完成 ({processed_count}): {os.path.basename(input_path)}")
        except Exception as e:
            result_queue.put(('error', input_path, str(e), None))
            print(f"[GPU {gpu_id}] 失败: {os.path.basename(input_path)} - {e}")
    
    print(f"[GPU {gpu_id}] 进程结束，共处理 {processed_count} 个视频")


def run_parallel(args, video_files, manifest_paths):
    """多 GPU 并行处理视频"""
    pending_videos = []
    for video_path in video_files:
        output_dir = get_output_dir(video_path, args.input, args.output)
        if is_processed(output_dir, manifest_paths):
            print(f"已跳过: {video_path}")
        else:
            pending_videos.append(video_path)
    
    if not pending_videos:
        print("所有视频已处理完成！")
        return
    
    print(f"\n待处理视频: {len(pending_videos)} 个")
    print(f"使用 GPU: {args.gpus}")
    
    gpu_list = [int(g) for g in args.gpus.split(',')]
    num_gpus = len(gpu_list)
    
    video_queue = mp.Queue()
    result_queue = mp.Queue()
    
    for video in pending_videos:
        video_queue.put(video)
    
    for _ in range(num_gpus):
        video_queue.put(None)
    
    processes = []
    for gpu_id in gpu_list:
        p = mp.Process(
            target=worker_process,
            args=(gpu_id, video_queue, result_queue, args.input, args.output)
        )
        p.start()
        processes.append(p)
    
    manifest_path = os.path.join(args.output, "segment_manifest.txt")
    empty_log_path = os.path.join(args.output, "empty_videos.log")
    
    success_count = 0
    error_count = 0
    
    while any(p.is_alive() for p in processes) or not result_queue.empty():
        try:
            result = result_queue.get(timeout=0.5)
        except:
            continue
        
        status, input_path, output_dir_or_error, segment_info = result
        
        if status == 'success':
            success_count += 1
            if segment_info:
                with open(manifest_path, "a") as f:
                    for seg_idx, (start_frame, end_frame) in enumerate(segment_info):
                        seg_path = os.path.abspath(os.path.join(output_dir_or_error, f"segment_{seg_idx:04d}.mp4"))
                        f.write(f"{seg_path} {start_frame} {end_frame}\n")
            else:
                with open(empty_log_path, "a") as ef:
                    ef.write(f"{input_path}\n")
        else:
            error_count += 1
            print(f"处理失败: {input_path} - {output_dir_or_error}")
    
    for p in processes:
        p.join()
    
    print(f"\n处理完成！成功: {success_count}, 失败: {error_count}")


def run_pipeline(args, video_files, manifest_paths):
    """
    流水线并行处理：
    1. 多线程预读取视频帧（CPU）
    2. GPU 批量特征提取
    3. 多线程保存视频（CPU）
    
    实现 CPU 和 GPU 操作的重叠，提升吞吐
    """
    from video_utils import video_to_frames
    from video_processing import _save_segments
    from uniseg_processor import UniSegProcessor
    from config import Config
    import numpy as np
    
    # 过滤已处理的视频
    pending_videos = []
    for video_path in video_files:
        output_dir = get_output_dir(video_path, args.input, args.output)
        if is_processed(output_dir, manifest_paths):
            print(f"已跳过: {video_path}")
        else:
            pending_videos.append((video_path, output_dir))
    
    if not pending_videos:
        print("所有视频已处理完成！")
        return
    
    print(f"\n待处理视频: {len(pending_videos)} 个")
    print(f"流水线模式: 预取={args.prefetch}, 保存线程={args.save_workers}")
    
    # 初始化模型
    print("\n初始化模型...")
    processor = UniSegProcessor()
    print("模型初始化完成\n")
    
    manifest_path = os.path.join(args.output, "segment_manifest.txt")
    empty_log_path = os.path.join(args.output, "empty_videos.log")
    
    # 预取队列
    prefetch_queue = Queue(maxsize=args.prefetch)
    # 保存队列
    save_queue = Queue()
    # 完成标志
    prefetch_done = threading.Event()
    save_done = threading.Event()
    
    # 统计
    stats = {'success': 0, 'error': 0, 'prefetch': 0, 'save': 0}
    stats_lock = threading.Lock()
    
    def prefetch_worker():
        """预读取线程：并行读取视频帧"""
        for video_path, output_dir in pending_videos:
            try:
                frames, fps, new_size = video_to_frames(video_path)
                prefetch_queue.put((video_path, output_dir, frames, fps, new_size, None))
                with stats_lock:
                    stats['prefetch'] += 1
                    print(f"[预取 {stats['prefetch']}/{len(pending_videos)}] {os.path.basename(video_path)}")
            except Exception as e:
                prefetch_queue.put((video_path, output_dir, None, None, None, str(e)))
        prefetch_done.set()
    
    def save_worker():
        """保存线程：并行保存分段视频"""
        while True:
            try:
                item = save_queue.get(timeout=1)
            except:
                if save_done.is_set() and save_queue.empty():
                    break
                continue
            
            if item is None:
                break
            
            video_path, output_dir, frames, fps, new_size, boundaries, total_frames = item
            
            try:
                os.makedirs(output_dir, exist_ok=True)
                frame_boundaries = _save_segments(frames, fps, new_size, boundaries, total_frames, output_dir)
                
                with stats_lock:
                    stats['save'] += 1
                    stats['success'] += 1
                
                # 写入 manifest
                if frame_boundaries:
                    with open(manifest_path, "a") as f:
                        for seg_idx, (start_frame, end_frame) in enumerate(frame_boundaries):
                            seg_path = os.path.abspath(os.path.join(output_dir, f"segment_{seg_idx:04d}.mp4"))
                            f.write(f"{seg_path} {start_frame} {end_frame}\n")
                else:
                    with open(empty_log_path, "a") as ef:
                        ef.write(f"{video_path}\n")
                
                print(f"[保存 {stats['save']}/{len(pending_videos)}] {os.path.basename(video_path)}")
                
            except Exception as e:
                with stats_lock:
                    stats['error'] += 1
                print(f"[保存失败] {os.path.basename(video_path)}: {e}")
    
    # 启动预取线程
    prefetch_thread = threading.Thread(target=prefetch_worker)
    prefetch_thread.start()
    
    # 启动保存线程池
    save_threads = []
    for _ in range(args.save_workers):
        t = threading.Thread(target=save_worker)
        t.start()
        save_threads.append(t)
    
    # GPU 处理主循环
    processed = 0
    while True:
        # 检查是否完成
        if prefetch_done.is_set() and prefetch_queue.empty():
            break
        
        try:
            item = prefetch_queue.get(timeout=1)
        except:
            continue
        
        video_path, output_dir, frames, fps, new_size, error = item
        
        if error:
            print(f"[读取失败] {os.path.basename(video_path)}: {error}")
            with stats_lock:
                stats['error'] += 1
            continue
        
        if frames is None or len(frames) == 0:
            print(f"[空视频] {os.path.basename(video_path)}")
            with open(empty_log_path, "a") as ef:
                ef.write(f"{video_path}\n")
            continue
        
        # GPU 特征提取和分割
        try:
            features = processor.extractor.extract_features(frames)
            G = processor.build_dynamic_graph(features, fps)
            from graph_operations import graph_propagation
            from boundary_detection import detect_boundaries
            G = graph_propagation(G)
            boundaries = detect_boundaries(G, fps)
            
            processed += 1
            print(f"[GPU {processed}/{len(pending_videos)}] {os.path.basename(video_path)} -> {len(boundaries)} 边界")
            
            # 提交到保存队列
            save_queue.put((video_path, output_dir, frames, fps, new_size, boundaries, len(frames)))
            
        except Exception as e:
            print(f"[GPU 处理失败] {os.path.basename(video_path)}: {e}")
            with stats_lock:
                stats['error'] += 1
    
    # 等待预取完成
    prefetch_thread.join()
    
    # 发送保存结束信号
    save_done.set()
    
    # 等待保存完成
    for t in save_threads:
        t.join()
    
    print(f"\n处理完成！成功: {stats['success']}, 失败: {stats['error']}")


def run_batch_pipeline(args, video_files, manifest_paths):
    """
    批量流水线：一次性加载多个视频的帧，批量处理特征
    适合视频较短的情况，可以显著提升 GPU 利用率
    """
    from video_utils import video_to_frames
    from video_processing import _save_segments
    from uniseg_processor import UniSegProcessor
    from config import Config
    import numpy as np
    import torch
    
    # 过滤已处理的视频
    pending_videos = []
    for video_path in video_files:
        output_dir = get_output_dir(video_path, args.input, args.output)
        if is_processed(output_dir, manifest_paths):
            print(f"已跳过: {video_path}")
        else:
            pending_videos.append((video_path, output_dir))
    
    if not pending_videos:
        print("所有视频已处理完成！")
        return
    
    print(f"\n待处理视频: {len(pending_videos)} 个")
    print(f"批量模式: batch_size={args.batch_size}")
    
    # 初始化模型
    print("\n初始化模型...")
    processor = UniSegProcessor()
    print("模型初始化完成\n")
    
    manifest_path = os.path.join(args.output, "segment_manifest.txt")
    empty_log_path = os.path.join(args.output, "empty_videos.log")
    
    success_count = 0
    error_count = 0
    
    # 按批次处理
    for batch_start in range(0, len(pending_videos), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(pending_videos))
        batch = pending_videos[batch_start:batch_end]
        
        print(f"\n--- 批次 {batch_start//args.batch_size + 1}/{(len(pending_videos) + args.batch_size - 1)//args.batch_size} ---")
        
        # 并行读取这批视频的帧
        batch_data = []
        with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
            def load_video(item):
                video_path, output_dir = item
                try:
                    frames, fps, new_size = video_to_frames(video_path)
                    return (video_path, output_dir, frames, fps, new_size, None)
                except Exception as e:
                    return (video_path, output_dir, None, None, None, str(e))
            
            results = list(executor.map(load_video, batch))
            batch_data = results
        
        print(f"  读取完成: {len(batch_data)} 个视频")
        
        # 逐个处理 GPU 特征提取（帧已在内存中）
        for video_path, output_dir, frames, fps, new_size, error in batch_data:
            if error:
                print(f"  [读取失败] {os.path.basename(video_path)}: {error}")
                error_count += 1
                continue
            
            if frames is None or len(frames) == 0:
                print(f"  [空视频] {os.path.basename(video_path)}")
                with open(empty_log_path, "a") as ef:
                    ef.write(f"{video_path}\n")
                continue
            
            try:
                # GPU 处理
                features = processor.extractor.extract_features(frames)
                G = processor.build_dynamic_graph(features, fps)
                from graph_operations import graph_propagation
                from boundary_detection import detect_boundaries
                G = graph_propagation(G)
                boundaries = detect_boundaries(G, fps)
                
                # 保存
                os.makedirs(output_dir, exist_ok=True)
                frame_boundaries = _save_segments(frames, fps, new_size, boundaries, len(frames), output_dir)
                
                if frame_boundaries:
                    with open(manifest_path, "a") as f:
                        for seg_idx, (start_frame, end_frame) in enumerate(frame_boundaries):
                            seg_path = os.path.abspath(os.path.join(output_dir, f"segment_{seg_idx:04d}.mp4"))
                            f.write(f"{seg_path} {start_frame} {end_frame}\n")
                else:
                    with open(empty_log_path, "a") as ef:
                        ef.write(f"{video_path}\n")
                
                success_count += 1
                print(f"  [完成 {success_count}] {os.path.basename(video_path)} -> {len(boundaries)} 边界")
                
            except Exception as e:
                error_count += 1
                print(f"  [失败] {os.path.basename(video_path)}: {e}")
        
        # 清理 GPU 内存
        torch.cuda.empty_cache()
    
    print(f"\n处理完成！成功: {success_count}, 失败: {error_count}")


def run_turbo(args, video_files, manifest_paths):
    """
    高吞吐模式：批量处理多个视频，最大化 GPU 利用率
    """
    from batch_processor import BatchVideoProcessor
    from config import Config
    
    manifest_path = os.path.join(args.output, "segment_manifest.txt")
    empty_log_path = os.path.join(args.output, "empty_videos.log")
    
    # 过滤已处理的视频
    pending_videos = []
    for video_path in video_files:
        output_dir = get_output_dir(video_path, args.input, args.output)
        if is_processed(output_dir, manifest_paths):
            pass  # 静默跳过
        else:
            pending_videos.append((video_path, output_dir))
    
    skip_count = len(video_files) - len(pending_videos)
    if skip_count > 0:
        print(f"跳过已处理: {skip_count} 个")
    
    if not pending_videos:
        print("所有视频已处理完成！")
        return
    
    print(f"待处理: {len(pending_videos)} 个视频")
    print(f"CLIP batch size: {Config.clip_batch_size}")
    print(f"视频 batch size: {args.batch_size}")
    
    # 初始化处理器
    processor = BatchVideoProcessor()
    
    success_count = 0
    error_count = 0
    
    # 批量处理
    batch_size = args.batch_size
    n_batches = (len(pending_videos) + batch_size - 1) // batch_size
    
    import time
    start_time = time.time()
    
    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(pending_videos))
        batch = pending_videos[batch_start:batch_end]
        
        print(f"\n{'='*60}")
        print(f"批次 {batch_idx + 1}/{n_batches} ({len(batch)} 视频)")
        print(f"{'='*60}")
        
        try:
            results = processor.process_video_batch(batch)
            
            # 保存结果到 manifest
            for video_path, output_dir, frame_boundaries, error in results:
                if error:
                    error_count += 1
                    print(f"  错误: {os.path.basename(video_path)} - {error}")
                elif frame_boundaries:
                    success_count += 1
                    with open(manifest_path, "a") as f:
                        for seg_idx, (start_frame, end_frame) in enumerate(frame_boundaries):
                            seg_path = os.path.abspath(os.path.join(output_dir, f"segment_{seg_idx:04d}.mp4"))
                            f.write(f"{seg_path} {start_frame} {end_frame}\n")
                else:
                    with open(empty_log_path, "a") as ef:
                        ef.write(f"{video_path}\n")
            
        except Exception as e:
            error_count += len(batch)
            print(f"  批次处理失败: {e}")
        
        # 进度统计
        elapsed = time.time() - start_time
        processed = batch_end
        remaining = len(pending_videos) - processed
        speed = processed / elapsed if elapsed > 0 else 0
        eta = remaining / speed if speed > 0 else 0
        
        print(f"  进度: {processed}/{len(pending_videos)}, 速度: {speed:.1f} 视频/秒, ETA: {eta/60:.1f} 分钟")
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"处理完成！")
    print(f"  成功: {success_count}")
    print(f"  失败: {error_count}")
    print(f"  跳过: {skip_count}")
    print(f"  总时间: {total_time/60:.1f} 分钟")
    print(f"  平均速度: {len(pending_videos)/total_time:.2f} 视频/秒")
    print(f"{'='*60}")


def run_serial(args, video_files, manifest_paths):
    """串行处理视频（单 GPU，共享模型实例）"""
    from video_processing import process_video_with_processor
    from uniseg_processor import UniSegProcessor
    
    manifest_path = os.path.join(args.output, "segment_manifest.txt")
    empty_log_path = os.path.join(args.output, "empty_videos.log")
    
    print("初始化模型...")
    processor = UniSegProcessor()
    print("模型初始化完成\n")
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for input_path in video_files:
        output_dir = get_output_dir(input_path, args.input, args.output)
        
        if is_processed(output_dir, manifest_paths):
            print(f"已跳过: {input_path}")
            skip_count += 1
            continue
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            segment_info = process_video_with_processor(processor, input_path, output_dir)
            
            if not segment_info:
                with open(empty_log_path, "a") as ef:
                    ef.write(f"{input_path}\n")
                print(f"无分割结果: {input_path}")
            else:
                with open(manifest_path, "a") as f:
                    for seg_idx, (start_frame, end_frame) in enumerate(segment_info):
                        seg_path = os.path.abspath(os.path.join(output_dir, f"segment_{seg_idx:04d}.mp4"))
                        f.write(f"{seg_path} {start_frame} {end_frame}\n")
                success_count += 1
                print(f"完成 ({success_count}): {input_path}")
        except Exception as e:
            error_count += 1
            print(f"处理失败: {input_path} - {str(e)}")
    
    print(f"\n处理完成！成功: {success_count}, 跳过: {skip_count}, 失败: {error_count}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./videos", help="输入视频目录路径")
    parser.add_argument("--output", default="./output", help="输出根目录路径")
    parser.add_argument("--gpus", default="0", help="使用的 GPU ID，逗号分隔（如 0,1）")
    parser.add_argument("--parallel", action="store_true", help="启用多 GPU 并行处理")
    parser.add_argument("--pipeline", action="store_true", help="启用流水线模式（CPU/GPU 重叠）")
    parser.add_argument("--batch", action="store_true", help="启用批量模式（预读取多个视频）")
    parser.add_argument("--batch_size", type=int, default=4, help="批量模式每批视频数")
    parser.add_argument("--prefetch", type=int, default=2, help="流水线模式预取队列大小")
    parser.add_argument("--save_workers", type=int, default=2, help="流水线模式保存线程数")
    parser.add_argument("--turbo", action="store_true", help="高吞吐模式（最大化GPU利用率）")
    args = parser.parse_args()
    
    # 创建输出目录和文件
    manifest_path = os.path.join(args.output, "segment_manifest.txt")
    empty_log_path = os.path.join(args.output, "empty_videos.log")
    os.makedirs(args.output, exist_ok=True)
    
    if not os.path.exists(manifest_path):
        with open(manifest_path, "w") as f:
            f.write("")
    if not os.path.exists(empty_log_path):
        with open(empty_log_path, "w") as ef:
            ef.write("empty_video_path\n")
    
    # 读取已处理的路径
    existing_seg_paths = set()
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            lines = f.readlines()
            if lines and lines[0].strip() == "file_path start_frame end_frame":
                for line in lines[1:]:
                    parts = line.strip().split()
                    if parts:
                        existing_seg_paths.add(parts[0])
    
    # 获取视频文件
    video_files = get_video_files(args.input)
    print(f"发现 {len(video_files)} 个视频文件")
    
    if not video_files:
        print("没有找到视频文件！")
        sys.exit(0)
    
    # 选择处理模式
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus.split(',')[0]
    
    if args.turbo:
        print("模式: 高吞吐（Turbo）")
        run_turbo(args, video_files, existing_seg_paths)
    elif args.parallel and ',' in args.gpus:
        print("模式: 多 GPU 并行")
        run_parallel(args, video_files, existing_seg_paths)
    elif args.pipeline:
        print("模式: 流水线（CPU/GPU 重叠）")
        run_pipeline(args, video_files, existing_seg_paths)
    elif args.batch:
        print("模式: 批量预读取")
        run_batch_pipeline(args, video_files, existing_seg_paths)
    else:
        print("模式: 串行")
        run_serial(args, video_files, existing_seg_paths)
    
    print(f"\n清单文件: {manifest_path}")
    print(f"空视频日志: {empty_log_path}")
