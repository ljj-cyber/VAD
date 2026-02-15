import os
import cv2
from config import Config
from video_utils import video_to_frames
from uniseg_processor import UniSegProcessor


def process_video_with_processor(processor, input_path, output_dir):
    """
    使用已初始化的 processor 处理视频（复用模型，避免重复初始化）
    """
    # 读取帧
    frames, fps, new_size = video_to_frames(input_path) 
    total_frames = len(frames)
    if total_frames == 0:
        return []
    
    # 处理视频获取边界
    boundaries = processor.process(frames, fps)
    
    return _save_segments(frames, fps, new_size, boundaries, total_frames, output_dir)


def process_video(input_path, output_dir):
    """
    处理单个视频（向后兼容，每次创建新的 processor）
    """
    processor = UniSegProcessor()
    
    # 读取帧
    frames, fps, new_size = video_to_frames(input_path) 
    total_frames = len(frames)
    if total_frames == 0:
        return []
    
    # 处理视频获取边界
    boundaries = processor.process(frames, fps)
    
    return _save_segments(frames, fps, new_size, boundaries, total_frames, output_dir)


def _save_segments(frames, fps, new_size, boundaries, total_frames, output_dir):
    """
    保存分段视频
    """

    # 转换边界为帧索引
    frame_boundaries = []
    if boundaries:
        for start_sec, end_sec in boundaries:
            start_frame = int(round(start_sec * fps))
            end_frame = min(int(round(end_sec * fps)), total_frames - 1)
            if end_frame - start_frame > 1:
                frame_boundaries.append((start_frame, end_frame))
        
        # 添加起始和结束边界
        if frame_boundaries and frame_boundaries[0][0] > 0:
            frame_boundaries.insert(0, (0, frame_boundaries[0][0]))
        
        last_end = frame_boundaries[-1][1] if frame_boundaries else 0
        if last_end < total_frames - 1:
            frame_boundaries.append((last_end, total_frames - 1))
    else:
        frame_boundaries.append((0, total_frames - 1))

    # 保存分段视频
    if frame_boundaries:
        codec_candidates = [
            ('mp4v', 'mp4v'),
            ('avc1', 'avc1'),
            ('xvid', 'XVID')
        ]
        
        selected_codec = None
        for codec_name, fourcc_code in codec_candidates:
            test_path = os.path.join(output_dir, f"test_{codec_name}.mp4")
            test_writer = cv2.VideoWriter(
                test_path, 
                cv2.VideoWriter_fourcc(*fourcc_code),
                fps, 
                new_size
            )
            if test_writer.isOpened():
                test_writer.release()
                os.remove(test_path)
                selected_codec = fourcc_code
                print(f"选择编码器: {fourcc_code}")
                break
        
        if selected_codec is None:
            print("无法找到新编码器，回退mp4v")
            selected_codec = 'mp4v'
        
        for seg_idx, (start, end) in enumerate(frame_boundaries):
            output_path = os.path.join(output_dir, f"segment_{seg_idx:04d}.mp4")
            
            out = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*selected_codec),
                fps,
                new_size
            )
            
            if not out.isOpened():
                print(f"视频写入失败，改用PNG序列保存片段 {seg_idx}")
                seg_dir = os.path.join(output_dir, f"segment_{seg_idx:04d}")
                os.makedirs(seg_dir, exist_ok=True)
                for idx in range(start, end + 1):
                    frame_bgr = frames[idx][:, :, ::-1]
                    cv2.imwrite(
                        os.path.join(seg_dir, f"frame_{idx:06d}.png"),
                        frame_bgr
                    )
                continue
                
            for idx in range(start, end + 1):
                frame_bgr = frames[idx][:, :, ::-1]
                out.write(frame_bgr)
            out.release()
    
    return frame_boundaries
