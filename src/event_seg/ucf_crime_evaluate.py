#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UCF Crime 分割效果评估脚本
功能：
1. 评估EventVAD分割与ground truth的对齐度
2. 计算异常检测相关指标
3. 生成评估报告
"""

import os
import json
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path


UCF_FPS = 30


def load_segment_manifest(manifest_path):
    """加载分割清单"""
    segments = []
    
    with open(manifest_path, 'r') as f:
        header = f.readline()  # 跳过头部
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 6:
                segments.append({
                    'segment_path': parts[0],
                    'start_frame': int(parts[1]),
                    'end_frame': int(parts[2]),
                    'category': parts[3],
                    'contains_anomaly': int(parts[4]) == 1,
                    'overlap_ratio': float(parts[5])
                })
    
    return segments


def load_annotations(annotation_file):
    """加载原始标注"""
    annotations = {}
    
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


def compute_boundary_metrics(segments_by_video, annotations):
    """
    计算边界检测指标
    
    评估EventVAD检测到的边界与标注的异常起止点的距离
    """
    boundary_errors = []
    detected_starts = []
    detected_ends = []
    
    for video_name, segments in segments_by_video.items():
        if video_name not in annotations:
            continue
        
        ann = annotations[video_name]
        if ann['is_normal']:
            continue
        
        # 获取分割边界
        seg_boundaries = []
        for seg in sorted(segments, key=lambda x: x['start_frame']):
            seg_boundaries.append(seg['start_frame'])
            seg_boundaries.append(seg['end_frame'])
        
        # 计算与标注边界的距离
        for anom_start, anom_end in ann['anomaly_frames']:
            # 找最近的分割边界
            if seg_boundaries:
                start_errors = [abs(b - anom_start) for b in seg_boundaries]
                end_errors = [abs(b - anom_end) for b in seg_boundaries]
                
                min_start_error = min(start_errors) / UCF_FPS  # 转换为秒
                min_end_error = min(end_errors) / UCF_FPS
                
                boundary_errors.append(min_start_error)
                boundary_errors.append(min_end_error)
                
                detected_starts.append(min_start_error < 1.0)  # 1秒内检测到
                detected_ends.append(min_end_error < 1.0)
    
    return {
        'mean_boundary_error': np.mean(boundary_errors) if boundary_errors else 0,
        'median_boundary_error': np.median(boundary_errors) if boundary_errors else 0,
        'boundary_detection_rate_1s': np.mean(detected_starts + detected_ends) if detected_starts else 0,
        'start_detection_rate': np.mean(detected_starts) if detected_starts else 0,
        'end_detection_rate': np.mean(detected_ends) if detected_ends else 0,
        'num_boundaries_evaluated': len(boundary_errors)
    }


def compute_segment_metrics(segments):
    """
    计算片段级别的指标
    """
    # 按类别统计
    category_stats = defaultdict(lambda: {
        'total': 0,
        'anomaly_segments': 0,
        'normal_segments': 0,
        'avg_overlap': []
    })
    
    for seg in segments:
        cat = seg['category']
        category_stats[cat]['total'] += 1
        
        if seg['contains_anomaly']:
            category_stats[cat]['anomaly_segments'] += 1
            category_stats[cat]['avg_overlap'].append(seg['overlap_ratio'])
        else:
            category_stats[cat]['normal_segments'] += 1
    
    # 计算统计
    results = {}
    for cat, stats in category_stats.items():
        results[cat] = {
            'total_segments': stats['total'],
            'anomaly_segments': stats['anomaly_segments'],
            'normal_segments': stats['normal_segments'],
            'anomaly_ratio': stats['anomaly_segments'] / stats['total'] if stats['total'] > 0 else 0,
            'avg_anomaly_overlap': np.mean(stats['avg_overlap']) if stats['avg_overlap'] else 0
        }
    
    return results


def compute_coverage_metrics(segments_by_video, annotations):
    """
    计算异常区间覆盖度
    
    评估EventVAD的分割是否完整覆盖了异常区间
    """
    coverage_scores = []
    fragmentation_scores = []
    
    for video_name, segments in segments_by_video.items():
        if video_name not in annotations:
            continue
        
        ann = annotations[video_name]
        if ann['is_normal']:
            continue
        
        for anom_start, anom_end in ann['anomaly_frames']:
            anom_length = anom_end - anom_start + 1
            
            # 计算被分割片段覆盖的部分
            covered_frames = set()
            covering_segments = 0
            
            for seg in segments:
                overlap_start = max(seg['start_frame'], anom_start)
                overlap_end = min(seg['end_frame'], anom_end)
                
                if overlap_end >= overlap_start:
                    for f in range(overlap_start, overlap_end + 1):
                        covered_frames.add(f)
                    covering_segments += 1
            
            coverage = len(covered_frames) / anom_length if anom_length > 0 else 0
            coverage_scores.append(coverage)
            
            # 碎片化程度（覆盖一个异常区间需要多少片段）
            if covering_segments > 0:
                fragmentation_scores.append(covering_segments)
    
    return {
        'mean_coverage': np.mean(coverage_scores) if coverage_scores else 0,
        'median_coverage': np.median(coverage_scores) if coverage_scores else 0,
        'full_coverage_rate': np.mean([c >= 0.99 for c in coverage_scores]) if coverage_scores else 0,
        'mean_fragmentation': np.mean(fragmentation_scores) if fragmentation_scores else 0,
        'num_anomalies_evaluated': len(coverage_scores)
    }


def evaluate_ucf_crime(manifest_path, annotation_file, output_path):
    """
    综合评估
    """
    print("加载数据...")
    segments = load_segment_manifest(manifest_path)
    annotations = load_annotations(annotation_file)
    
    print(f"加载了 {len(segments)} 个片段")
    print(f"加载了 {len(annotations)} 条标注")
    
    # 按视频分组
    segments_by_video = defaultdict(list)
    for seg in segments:
        # 从路径提取视频名
        video_name = os.path.basename(os.path.dirname(seg['segment_path']))
        if video_name.endswith('_segments'):
            video_name = video_name[:-9] + '.mp4'
        segments_by_video[video_name].append(seg)
    
    print(f"涉及 {len(segments_by_video)} 个视频")
    
    # 计算各项指标
    print("\n计算边界检测指标...")
    boundary_metrics = compute_boundary_metrics(segments_by_video, annotations)
    
    print("计算片段统计...")
    segment_metrics = compute_segment_metrics(segments)
    
    print("计算覆盖度指标...")
    coverage_metrics = compute_coverage_metrics(segments_by_video, annotations)
    
    # 汇总结果
    results = {
        'summary': {
            'total_segments': len(segments),
            'total_videos': len(segments_by_video),
            'total_annotations': len(annotations)
        },
        'boundary_metrics': boundary_metrics,
        'coverage_metrics': coverage_metrics,
        'segment_metrics': segment_metrics
    }
    
    # 保存结果
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 打印报告
    print("\n" + "="*60)
    print("UCF Crime EventVAD 评估报告")
    print("="*60)
    
    print("\n【边界检测指标】")
    print(f"  平均边界误差: {boundary_metrics['mean_boundary_error']:.2f} 秒")
    print(f"  中位边界误差: {boundary_metrics['median_boundary_error']:.2f} 秒")
    print(f"  1秒内检测率: {boundary_metrics['boundary_detection_rate_1s']*100:.1f}%")
    print(f"  起始边界检测率: {boundary_metrics['start_detection_rate']*100:.1f}%")
    print(f"  结束边界检测率: {boundary_metrics['end_detection_rate']*100:.1f}%")
    
    print("\n【覆盖度指标】")
    print(f"  平均覆盖率: {coverage_metrics['mean_coverage']*100:.1f}%")
    print(f"  中位覆盖率: {coverage_metrics['median_coverage']*100:.1f}%")
    print(f"  完全覆盖率: {coverage_metrics['full_coverage_rate']*100:.1f}%")
    print(f"  平均碎片化程度: {coverage_metrics['mean_fragmentation']:.2f} 片段/异常")
    
    print("\n【各类别片段统计】")
    for cat in sorted(segment_metrics.keys()):
        stats = segment_metrics[cat]
        print(f"  {cat}:")
        print(f"    总片段数: {stats['total_segments']}")
        print(f"    异常片段: {stats['anomaly_segments']} ({stats['anomaly_ratio']*100:.1f}%)")
        print(f"    平均重叠度: {stats['avg_anomaly_overlap']*100:.1f}%")
    
    print(f"\n评估结果已保存到: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='UCF Crime 分割效果评估')
    parser.add_argument('--manifest', type=str,
                        default='./output/ucf_crime/segment_manifest.txt',
                        help='分割清单文件路径')
    parser.add_argument('--annotation', type=str,
                        default='./videos/ucf_crime/Temporal_Anomaly_Annotation_for_Testing_Videos.txt',
                        help='原始标注文件路径')
    parser.add_argument('--output', type=str,
                        default='./output/ucf_crime/evaluation_results.json',
                        help='评估结果输出路径')
    
    args = parser.parse_args()
    
    evaluate_ucf_crime(args.manifest, args.annotation, args.output)


if __name__ == '__main__':
    main()
