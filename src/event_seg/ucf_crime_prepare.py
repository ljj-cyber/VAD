#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UCF Crime 数据集准备脚本
功能：
1. 解压数据集zip文件
2. 组织视频文件结构
3. 解析时间标注文件
"""

import os
import zipfile
import shutil
import argparse
from pathlib import Path


# UCF Crime 异常类别
ANOMALY_CATEGORIES = [
    'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 
    'Explosion', 'Fighting', 'RoadAccidents', 'Robbery', 
    'Shooting', 'Shoplifting', 'Stealing', 'Vandalism'
]


def extract_zip_files(ucf_crime_dir, output_dir):
    """
    解压UCF Crime数据集的所有zip文件
    
    Args:
        ucf_crime_dir: UCF Crime数据集目录（包含zip文件）
        output_dir: 解压输出目录
    """
    ucf_crime_path = Path(ucf_crime_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 定义需要解压的zip文件
    zip_files = [
        # 异常视频
        'Anomaly-Videos-Part-1.zip',
        'Anomaly-Videos-Part-2.zip',
        'Anomaly-Videos-Part-3.zip',
        'Anomaly-Videos-Part-4.zip',
        # 训练正常视频
        'Training-Normal-Videos-Part-1.zip',
        'Training-Normal-Videos-Part-2.zip',
        # 测试正常视频
        'Testing_Normal_Videos.zip',
    ]
    
    for zip_name in zip_files:
        zip_path = ucf_crime_path / zip_name
        if zip_path.exists():
            print(f"正在解压: {zip_name}")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(output_path)
                print(f"完成解压: {zip_name}")
            except Exception as e:
                print(f"解压失败 {zip_name}: {e}")
        else:
            print(f"未找到: {zip_name}")
    
    return output_path


def parse_temporal_annotations(annotation_file):
    """
    解析时间标注文件
    
    Args:
        annotation_file: 时间标注文件路径
        
    Returns:
        dict: {视频名: {'category': 类别, 'anomaly_frames': [(start, end), ...]}}
    """
    annotations = {}
    
    with open(annotation_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            
            video_name = parts[0]
            category = parts[1]
            start1, end1 = int(parts[2]), int(parts[3])
            start2, end2 = int(parts[4]), int(parts[5])
            
            anomaly_frames = []
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


def organize_videos(extracted_dir, organized_dir):
    """
    组织解压后的视频文件，按类别放置
    
    Args:
        extracted_dir: 解压后的目录
        organized_dir: 组织后的目录
    """
    extracted_path = Path(extracted_dir)
    organized_path = Path(organized_dir)
    
    # 创建目录结构
    train_anomaly_dir = organized_path / 'train' / 'anomaly'
    train_normal_dir = organized_path / 'train' / 'normal'
    test_anomaly_dir = organized_path / 'test' / 'anomaly'
    test_normal_dir = organized_path / 'test' / 'normal'
    
    for d in [train_anomaly_dir, train_normal_dir, test_anomaly_dir, test_normal_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # 统计信息
    stats = {'train_anomaly': 0, 'train_normal': 0, 'test_anomaly': 0, 'test_normal': 0}
    
    # 遍历所有mp4文件
    for mp4_file in extracted_path.rglob('*.mp4'):
        video_name = mp4_file.name
        parent_dir = mp4_file.parent.name
        
        # 判断类别和训练/测试集
        category = None
        for cat in ANOMALY_CATEGORIES:
            if video_name.startswith(cat):
                category = cat
                break
        
        is_anomaly = category is not None
        is_training = 'Training' in str(mp4_file) or parent_dir in ANOMALY_CATEGORIES
        
        # 确定目标目录
        if is_anomaly:
            if is_training:
                target_dir = train_anomaly_dir / category
                stats['train_anomaly'] += 1
            else:
                target_dir = test_anomaly_dir / category
                stats['test_anomaly'] += 1
        else:
            if 'Testing' in str(mp4_file):
                target_dir = test_normal_dir
                stats['test_normal'] += 1
            else:
                target_dir = train_normal_dir
                stats['train_normal'] += 1
        
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / video_name
        
        # 复制文件（如果不存在）
        if not target_path.exists():
            shutil.copy2(mp4_file, target_path)
            print(f"复制: {video_name} -> {target_dir}")
    
    print("\n=== 视频组织统计 ===")
    print(f"训练异常视频: {stats['train_anomaly']}")
    print(f"训练正常视频: {stats['train_normal']}")
    print(f"测试异常视频: {stats['test_anomaly']}")
    print(f"测试正常视频: {stats['test_normal']}")
    
    return stats


def create_video_list(organized_dir, annotation_file, output_file):
    """
    创建视频列表文件，包含标注信息
    
    Args:
        organized_dir: 组织后的视频目录
        annotation_file: 原始标注文件
        output_file: 输出文件路径
    """
    organized_path = Path(organized_dir)
    
    # 解析标注
    annotations = parse_temporal_annotations(annotation_file)
    
    with open(output_file, 'w') as f:
        f.write("video_path\tcategory\tis_anomaly\tanomaly_start1\tanomaly_end1\tanomaly_start2\tanomaly_end2\n")
        
        for mp4_file in sorted(organized_path.rglob('*.mp4')):
            video_name = mp4_file.name
            rel_path = mp4_file.relative_to(organized_path)
            
            # 获取标注信息
            if video_name in annotations:
                ann = annotations[video_name]
                category = ann['category']
                is_anomaly = 0 if ann['is_normal'] else 1
                
                if ann['anomaly_frames']:
                    start1, end1 = ann['anomaly_frames'][0]
                    if len(ann['anomaly_frames']) > 1:
                        start2, end2 = ann['anomaly_frames'][1]
                    else:
                        start2, end2 = -1, -1
                else:
                    start1, end1, start2, end2 = -1, -1, -1, -1
            else:
                # 根据路径推断
                for cat in ANOMALY_CATEGORIES:
                    if cat in str(mp4_file):
                        category = cat
                        is_anomaly = 1
                        break
                else:
                    category = 'Normal'
                    is_anomaly = 0
                start1, end1, start2, end2 = -1, -1, -1, -1
            
            f.write(f"{rel_path}\t{category}\t{is_anomaly}\t{start1}\t{end1}\t{start2}\t{end2}\n")
    
    print(f"\n视频列表已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='UCF Crime 数据集准备工具')
    parser.add_argument('--ucf_dir', type=str, 
                        default='./videos/ucf_crime',
                        help='UCF Crime数据集目录（包含zip文件）')
    parser.add_argument('--extract_dir', type=str,
                        default='./videos/ucf_crime_extracted',
                        help='解压输出目录')
    parser.add_argument('--organized_dir', type=str,
                        default='./videos/ucf_crime_organized',
                        help='组织后的视频目录')
    parser.add_argument('--skip_extract', action='store_true',
                        help='跳过解压步骤')
    parser.add_argument('--skip_organize', action='store_true',
                        help='跳过组织步骤')
    
    args = parser.parse_args()
    
    ucf_dir = Path(args.ucf_dir).resolve()
    extract_dir = Path(args.extract_dir).resolve()
    organized_dir = Path(args.organized_dir).resolve()
    
    print("=== UCF Crime 数据集准备 ===")
    print(f"UCF Crime目录: {ucf_dir}")
    print(f"解压目录: {extract_dir}")
    print(f"组织目录: {organized_dir}")
    
    # 1. 解压文件
    if not args.skip_extract:
        print("\n[步骤1] 解压数据集文件...")
        extract_zip_files(ucf_dir, extract_dir)
    else:
        print("\n[步骤1] 跳过解压")
    
    # 2. 组织视频
    if not args.skip_organize:
        print("\n[步骤2] 组织视频文件...")
        organize_videos(extract_dir, organized_dir)
    else:
        print("\n[步骤2] 跳过组织")
    
    # 3. 创建视频列表
    print("\n[步骤3] 创建视频列表...")
    annotation_file = ucf_dir / 'Temporal_Anomaly_Annotation_for_Testing_Videos.txt'
    video_list_file = organized_dir / 'video_list.txt'
    
    if annotation_file.exists():
        create_video_list(organized_dir, annotation_file, video_list_file)
    else:
        print(f"警告: 未找到标注文件 {annotation_file}")
    
    print("\n=== 准备完成 ===")
    print(f"视频目录: {organized_dir}")
    print(f"视频列表: {video_list_file}")


if __name__ == '__main__':
    main()
