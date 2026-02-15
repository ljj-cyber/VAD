import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from config import Config


def video_to_frames(video_path, num_workers=None, sample_rate=None):
    """
    优化的视频帧读取
    
    Args:
        video_path: 视频路径
        num_workers: 多线程数量
        sample_rate: 帧采样率（每 N 帧取 1 帧）
    
    Returns:
        frames: 帧数组
        fps: 有效帧率（考虑采样后的）
        new_size: 帧尺寸
    """
    if num_workers is None:
        num_workers = Config.num_workers
    if sample_rate is None:
        sample_rate = getattr(Config, 'frame_sample_rate', 1)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件：{video_path}")
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 计算目标分辨率
    ratio = min(Config.max_resolution[0]/width, Config.max_resolution[1]/height)
    new_size = (int(width*ratio), int(height*ratio)) if ratio < 1 else (width, height)
    # 确保尺寸是 8 的倍数（RAFT 要求）
    new_size = (new_size[0]//8*8, new_size[1]//8*8)
    
    # 按采样率读取帧
    raw_frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 按采样率选取帧
        if frame_idx % sample_rate == 0:
            raw_frames.append(frame)
        
        frame_idx += 1
    
    cap.release()
    
    if not raw_frames:
        return np.array([]), original_fps / sample_rate, new_size
    
    # 多线程处理帧（resize + 颜色转换）
    def process_frame(frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, new_size)
        return frame.astype(np.uint8)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        frames = list(executor.map(process_frame, raw_frames))
    
    # 返回有效帧率（采样后的）
    effective_fps = original_fps / sample_rate
    
    if sample_rate > 1:
        print(f"读取 {len(frames)} 帧（采样率 1/{sample_rate}，原始 {total_frames} 帧）")
    else:
        print(f"读取 {len(frames)} 帧")
    
    return np.array(frames), effective_fps, new_size


def video_to_frames_fast(video_path, max_frames=500, sample_rate=None):
    """
    快速视频读取 - 限制最大帧数
    适合大规模处理
    """
    if sample_rate is None:
        sample_rate = getattr(Config, 'frame_sample_rate', 1)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件：{video_path}")
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 如果视频太长，增加采样率
    expected_frames = total_frames // sample_rate
    if expected_frames > max_frames:
        sample_rate = max(sample_rate, total_frames // max_frames)
    
    # 计算目标分辨率
    ratio = min(Config.max_resolution[0]/width, Config.max_resolution[1]/height)
    new_size = (int(width*ratio), int(height*ratio)) if ratio < 1 else (width, height)
    new_size = (new_size[0]//8*8, new_size[1]//8*8)
    
    frames = []
    frame_idx = 0
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_rate == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, new_size)
            frames.append(frame.astype(np.uint8))
        
        frame_idx += 1
    
    cap.release()
    
    effective_fps = original_fps / sample_rate
    
    print(f"快速读取 {len(frames)} 帧（采样率 1/{sample_rate}）")
    
    return np.array(frames) if frames else np.array([]), effective_fps, new_size


def video_to_frames_decord(video_path, sample_rate=None):
    """
    使用 decord 库进行 GPU 加速的帧读取
    """
    if sample_rate is None:
        sample_rate = getattr(Config, 'frame_sample_rate', 1)
    
    try:
        from decord import VideoReader, cpu, gpu
        
        try:
            vr = VideoReader(video_path, ctx=gpu(0))
        except:
            vr = VideoReader(video_path, ctx=cpu(0))
        
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        
        sample_frame = vr[0].asnumpy()
        height, width = sample_frame.shape[:2]
        
        ratio = min(Config.max_resolution[0]/width, Config.max_resolution[1]/height)
        new_size = (int(width*ratio), int(height*ratio)) if ratio < 1 else (width, height)
        new_size = (new_size[0]//8*8, new_size[1]//8*8)
        
        # 按采样率读取
        indices = list(range(0, total_frames, sample_rate))
        frames = vr.get_batch(indices).asnumpy()
        
        # resize
        if new_size != (width, height):
            resized_frames = []
            for frame in frames:
                resized_frames.append(cv2.resize(frame, new_size))
            frames = np.array(resized_frames)
        
        effective_fps = fps / sample_rate
        print(f"读取 {len(frames)} 帧（decord, 采样率 1/{sample_rate}）")
        
        return frames, effective_fps, new_size
        
    except ImportError:
        return video_to_frames(video_path, sample_rate=sample_rate)
