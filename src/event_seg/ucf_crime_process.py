#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UCF Crime 数据集处理脚本 - 适配 EventVAD
功能：
1. 使用EventVAD进行视频事件分割
2. 生成带标注的分割清单
3. 计算分割与标注的重叠度
"""

import os
import sys
import pathlib
import argparse
import glob
import json
from collections import defaultdict

# 设置路径
_current_dir = pathlib.Path(__file__).parent.resolve()
_src_dir = _current_dir.parent.resolve()
_project_root = _src_dir.parent.resolve()

sys.path.insert(0, str(_current_dir))
sys.path.insert(0, str(_src_dir))
sys.path.insert(0, str(_src_dir / "LAVIS"))

from video_processing import process_video

# UCF Crime 常量
UCF_FPS = 30  # UCF Crime 视频帧率
ANOMALY_CATEGORIES = [
    'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 
    'Explosion', 'Fighting', 'RoadAccidents', 'Robbery', 
    'Shooting', 'Shoplifting', 'Stealing', 'Vandalism'
]


def parse_annotation_file(annotation_file):
    """
    解析UCF Crime时间标注文件
    
    Returns:
        dict: {视频名: {'category': str, 'anomaly_frames': [(start, end), ...]}}
    """
    annotations = {}
    
    if not os.path.exists(annotation_file):
        return annotations
    
    with open(annotation_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            
            video_name = parts[0]
            category = parts[1]
            
            anomaly_frames = []
            start1, end1 = int(parts[2]), int(parts[3])
            start2, end2 = int(parts[4]), int(parts[5])
            
            if start1 >= 0 and end1 >= 0:
                anomaly_frames.append((start1, end1))
            if start2 >= 0 and end2 >= 0:
                anomaly_frames.append((start2, end2))
            
            annotations[video_name] = {
                'category': category,
                'anomaly_frames': anomaly_frames,
                'is_normal': category == 'Normal' or len(anomaly_frames) == 0
            }
    
    return annotations


def get_category_from_path(video_path):
    """从视频路径推断类别"""
    video_name = os.path.basename(video_path)
    for cat in ANOMALY_CATEGORIES:
        if video_name.startswith(cat):
            return cat
    if 'Normal' in video_name or 'normal' in video_path.lower():
        return 'Normal'
    return 'Unknown'


def compute_overlap(seg_start, seg_end, anomaly_ranges):
    """
    计算分割片段与异常区间的重叠度
    
    Args:
        seg_start: 片段起始帧
        seg_end: 片段结束帧
        anomaly_ranges: 异常帧范围列表 [(start, end), ...]
    
    Returns:
        tuple: (是否包含异常, 重叠比例)
    """
    if not anomaly_ranges:
        return False, 0.0
    
    seg_length = seg_end - seg_start + 1
    total_overlap = 0
    
    for anom_start, anom_end in anomaly_ranges:
        overlap_start = max(seg_start, anom_start)
        overlap_end = min(seg_end, anom_end)
        
        if overlap_end >= overlap_start:
            total_overlap += overlap_end - overlap_start + 1
    
    overlap_ratio = total_overlap / seg_length if seg_length > 0 else 0
    contains_anomaly = total_overlap > 0
    
    return contains_anomaly, overlap_ratio


def process_ucf_crime_video(input_path, output_dir, annotation=None):
    """
    处理单个UCF Crime视频
    
    Args:
        input_path: 输入视频路径
        output_dir: 输出目录
        annotation: 视频标注信息（可选）
    
    Returns:
        list: 分割信息列表 [(seg_path, start_frame, end_frame, contains_anomaly, overlap_ratio), ...]
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 使用EventVAD进行分割
        segment_info = process_video(input_path, output_dir)
        
        if not segment_info:
            return None
        
        # 处理分割结果，添加标注信息
        results = []
        anomaly_frames = annotation.get('anomaly_frames', []) if annotation else []
        
        for seg_idx, (start_frame, end_frame) in enumerate(segment_info):
            seg_path = os.path.join(output_dir, f"segment_{seg_idx:04d}.mp4")
            
            # 计算与异常区间的重叠
            contains_anomaly, overlap_ratio = compute_overlap(
                start_frame, end_frame, anomaly_frames
            )
            
            results.append({
                'segment_path': seg_path,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'start_time': start_frame / UCF_FPS,
                'end_time': end_frame / UCF_FPS,
                'contains_anomaly': contains_anomaly,
                'anomaly_overlap_ratio': overlap_ratio
            })
        
        return results
    
    except Exception as e:
        print(f"处理失败: {input_path} - {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(description='UCF Crime 数据集处理 - EventVAD')
    parser.add_argument('--input', type=str, 
                        default='./videos/ucf_crime_organized',
                        help='输入视频目录路径')
    parser.add_argument('--output', type=str,
                        default='./output/ucf_crime',
                        help='输出根目录路径')
    parser.add_argument('--annotation', type=str,
                        default='./videos/ucf_crime/Temporal_Anomaly_Annotation_for_Testing_Videos.txt',
                        help='时间标注文件路径')
    parser.add_argument('--subset', type=str, default='all',
                        choices=['all', 'train', 'test', 'anomaly', 'normal'],
                        help='处理的数据子集')
    parser.add_argument('--category', type=str, default=None,
                        help='只处理指定类别（如 Robbery, Shooting 等）')
    parser.add_argument('--gpu', type=str, default='0',
                        help='使用的GPU编号')
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # 解析标注文件
    print(f"加载标注文件: {args.annotation}")
    annotations = parse_annotation_file(args.annotation)
    print(f"加载了 {len(annotations)} 条标注")
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 输出文件
    manifest_path = os.path.join(args.output, "segment_manifest.txt")
    annotation_manifest_path = os.path.join(args.output, "segment_manifest_annotated.json")
    empty_log_path = os.path.join(args.output, "empty_videos.log")
    stats_path = os.path.join(args.output, "processing_stats.json")
    
    # 初始化文件
    if not os.path.exists(manifest_path):
        with open(manifest_path, "w") as f:
            f.write("segment_path\tstart_frame\tend_frame\tcategory\tcontains_anomaly\toverlap_ratio\n")
    
    if not os.path.exists(empty_log_path):
        with open(empty_log_path, "w") as ef:
            ef.write("empty_video_path\n")
    
    # 加载已处理的视频
    processed_videos = set()
    if os.path.exists(annotation_manifest_path):
        with open(annotation_manifest_path, 'r') as f:
            existing_data = json.load(f)
            processed_videos = set(existing_data.get('processed_videos', []))
    
    # 收集所有视频文件
    video_files = []
    for root, dirs, files in os.walk(args.input):
        for file in files:
            if file.split('.')[-1].lower() in ['mp4', 'avi', 'mov']:
                video_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, args.input)
                
                # 过滤子集
                if args.subset != 'all':
                    if args.subset in ['train', 'test'] and args.subset not in rel_path:
                        continue
                    if args.subset == 'anomaly' and 'normal' in rel_path.lower():
                        continue
                    if args.subset == 'normal' and 'anomaly' in rel_path.lower():
                        continue
                
                # 过滤类别
                if args.category:
                    category = get_category_from_path(video_path)
                    if category != args.category:
                        continue
                
                video_files.append((video_path, file, rel_path))
    
    print(f"\n找到 {len(video_files)} 个视频文件")
    
    # 统计信息
    stats = {
        'total_videos': len(video_files),
        'processed': 0,
        'skipped': 0,
        'failed': 0,
        'empty': 0,
        'total_segments': 0,
        'anomaly_segments': 0,
        'categories': defaultdict(int)
    }
    
    # 处理结果收集
    all_results = {
        'processed_videos': list(processed_videos),
        'segments': []
    }
    
    # 处理每个视频
    for idx, (video_path, video_name, rel_path) in enumerate(video_files):
        print(f"\n[{idx+1}/{len(video_files)}] 处理: {video_name}")
        
        # 跳过已处理的
        if video_path in processed_videos:
            print(f"  已跳过（已处理）")
            stats['skipped'] += 1
            continue
        
        # 获取类别和标注
        category = get_category_from_path(video_path)
        annotation = annotations.get(video_name, None)
        
        # 构建输出目录
        video_base = os.path.splitext(video_name)[0]
        output_dir = os.path.join(args.output, rel_path, f"{video_base}_segments")
        
        # 检查是否已有输出
        if os.path.isdir(output_dir):
            existing_segments = glob.glob(os.path.join(output_dir, "segment_*.mp4"))
            if len(existing_segments) > 0:
                print(f"  已跳过（已存在输出）")
                stats['skipped'] += 1
                processed_videos.add(video_path)
                continue
        
        # 处理视频
        results = process_ucf_crime_video(video_path, output_dir, annotation)
        
        if results is None:
            with open(empty_log_path, "a") as ef:
                ef.write(f"{video_path}\n")
            print(f"  无分割结果")
            stats['empty'] += 1
            continue
        
        # 记录结果
        stats['processed'] += 1
        stats['total_segments'] += len(results)
        stats['categories'][category] += 1
        
        processed_videos.add(video_path)
        all_results['processed_videos'].append(video_path)
        
        for seg_info in results:
            seg_info['source_video'] = video_path
            seg_info['category'] = category
            all_results['segments'].append(seg_info)
            
            if seg_info['contains_anomaly']:
                stats['anomaly_segments'] += 1
            
            # 写入清单文件
            with open(manifest_path, "a") as f:
                f.write(f"{seg_info['segment_path']}\t")
                f.write(f"{seg_info['start_frame']}\t")
                f.write(f"{seg_info['end_frame']}\t")
                f.write(f"{category}\t")
                f.write(f"{1 if seg_info['contains_anomaly'] else 0}\t")
                f.write(f"{seg_info['anomaly_overlap_ratio']:.4f}\n")
        
        print(f"  完成: {len(results)} 个片段")
        
        # 定期保存
        if (idx + 1) % 10 == 0:
            with open(annotation_manifest_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            with open(stats_path, 'w') as f:
                json.dump(dict(stats), f, indent=2)
    
    # 最终保存
    with open(annotation_manifest_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    stats['categories'] = dict(stats['categories'])
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # 打印统计
    print("\n" + "="*50)
    print("UCF Crime 处理完成")
    print("="*50)
    print(f"总视频数: {stats['total_videos']}")
    print(f"已处理: {stats['processed']}")
    print(f"已跳过: {stats['skipped']}")
    print(f"失败: {stats['failed']}")
    print(f"空结果: {stats['empty']}")
    print(f"总片段数: {stats['total_segments']}")
    print(f"异常片段数: {stats['anomaly_segments']}")
    print(f"\n类别统计:")
    for cat, count in sorted(stats['categories'].items()):
        print(f"  {cat}: {count}")
    print(f"\n输出文件:")
    print(f"  清单文件: {manifest_path}")
    print(f"  标注清单: {annotation_manifest_path}")
    print(f"  统计信息: {stats_path}")


if __name__ == '__main__':
    main()
