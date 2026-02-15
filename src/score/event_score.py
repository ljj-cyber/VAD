import os
import sys
import pathlib
import json
import re
import multiprocessing as mp

# 设置 Hugging Face 模型缓存目录到项目目录（必须在导入 transformers 相关库之前设置）
_current_dir = pathlib.Path(__file__).parent.resolve()
_project_root = _current_dir.parent.parent.resolve()
_hf_cache_dir = _project_root / "models" / "huggingface"
os.environ["HF_HOME"] = str(_hf_cache_dir)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(_hf_cache_dir / "hub")

# 禁用 Flash Attention 2.0（SiglipVisionModel 不支持）
os.environ["ATTN_BACKEND"] = "eager"

import csv
import torch
from pathlib import Path
from tqdm import tqdm
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init

NUM_FRAMES = 16
BATCH_SIZE = 4

# Video Anomaly Detection Prompt Template (用户提供版本)
PROMPT_TEMPLATE = """
### Role
You are acting as a law enforcement agency specialized in Video Anomaly Detection (VAD).

### Context
How would you rate the scene described below on a scale from 0 to 1, with 0 representing a standard scene and 1 denoting a scene with suspicious or potentially criminal activities?

### Task
1. **Summary**: Summarize what happened in a few sentences based on the following temporal description. Do not include unnecessary details.
2. **Reasoning (CoT)**: Perform a step-by-step Chain-of-Thought analysis to evaluate the level of suspicion or criminality.
3. **Scoring**: Assign a single score from this list: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0].

### Input Data
{temporal_description}

### Output Format
Respond STRICTLY in JSON format:
{{
  "summary": "Your brief summary",
  "reasoning": "Your step-by-step CoT analysis",
  "score": [0.0] 
}}
"""


def _parse_score_value(score_value):
    if isinstance(score_value, list) and score_value:
        score_value = score_value[0]
    return float(score_value)


def extract_score_from_response(response):
    """从模型响应中提取分数"""
    # 尝试解析 JSON
    try:
        # 查找 JSON 块
        json_match = re.search(r'\{[^{}]*"score"[^{}]*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            data = json.loads(json_str)
            score = _parse_score_value(data.get('score', -1))
            if 0 <= score <= 1:
                return score, "success", data.get('summary', ''), data.get('reasoning', '')
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    
    # 备用方案：直接查找分数
    score_patterns = [
        r'"score"\s*:\s*(\d+\.?\d*)',
        r'score[:\s]+(\d+\.?\d*)',
        r'\b(0\.\d+|1\.0|0|1)\b'
    ]
    
    for pattern in score_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                if 0 <= score <= 1:
                    return score, "success", "", ""
            except ValueError:
                continue
    
    return -1, f"Error: Cannot parse score from response - {response[:200]}", "", ""


def init_model(device):
    disable_torch_init()
    model_path = 'DAMO-NLP-SG/VideoLLaMA2.1-7B-16F'
    
    # 禁用 Flash Attention 2.0（SiglipVisionModel 不支持）
    model, processor, tokenizer = model_init(
        model_path, 
        device_map=device,
        attn_implementation="eager"  # 使用标准注意力实现
    )
    
    return model.half().eval(), processor, tokenizer


def process_files(video_files, output_csv, device="cuda:0"):
    # 性能优化
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    model, processor, tokenizer = init_model(device)
    
    with open(output_csv, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'score', 'status', 'summary', 'reasoning'])
        
        for path_str in tqdm(video_files, desc=f"Processing videos on {device}"):
            path = Path(path_str)
            if not path.exists():
                writer.writerow([str(path), -1, "Error: File not found", "", ""])
                continue
            try:
                video_input = processor['video'](str(path)).to(device)
                
                with torch.cuda.amp.autocast(), torch.no_grad():
                    temporal_description = "No text description is provided. Please infer from the video content."
                    prompt = PROMPT_TEMPLATE.format(temporal_description=temporal_description)
                    output = mm_infer(
                        video_input,
                        prompt,
                        model=model,
                        tokenizer=tokenizer,
                        do_sample=False,
                        modal='video'
                    )
                
                score, status, summary, reasoning = extract_score_from_response(output)
                writer.writerow([str(path), score, status, summary, reasoning])
                
            except Exception as e:
                writer.writerow([str(path), -1, f"Error: {str(e)}", "", ""])
            finally:
                torch.cuda.empty_cache()


def split_list(items, num_chunks):
    chunks = [[] for _ in range(num_chunks)]
    for i, item in enumerate(items):
        chunks[i % num_chunks].append(item)
    return chunks


def _worker_process(video_files, output_csv, device):
    os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1]
    process_files(video_files, output_csv, device="cuda:0")


def main(input_csv, output_csv, gpus="0"):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Manifest文件不存在: {input_csv}")
    
    output_dir = os.path.dirname(output_csv)
    if output_dir:
    os.makedirs(output_dir, exist_ok=True)

    original_entries = []
    video_files = []
    with open(input_csv, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                path = parts[0]
                start = parts[1]
                end = parts[2]
                video_files.append(path)
                original_entries.append({
                    'path': path,
                    'start': start,
                    'end': end,
                    'score': None
                })
    
    gpu_list = [g.strip() for g in gpus.split(",") if g.strip() != ""]
    if len(gpu_list) <= 1:
        if gpu_list:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list[0]
        process_files(video_files, output_csv, device="cuda:0")
    else:
        # 多 GPU 并行推理
        tmp_dir = os.path.join(output_dir if output_dir else ".", "tmp_scores")
        os.makedirs(tmp_dir, exist_ok=True)
        
        chunks = split_list(video_files, len(gpu_list))
        processes = []
        tmp_files = []
        
        for idx, (gpu_id, chunk) in enumerate(zip(gpu_list, chunks)):
            if not chunk:
                continue
            tmp_csv = os.path.join(tmp_dir, f"scores_gpu{gpu_id}.csv")
            tmp_files.append(tmp_csv)
            p = mp.get_context("spawn").Process(
                target=_worker_process,
                args=(chunk, tmp_csv, f"cuda:{gpu_id}")
            )
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
        
        # 合并临时结果
        scores_dict = {}
        for tmp_csv in tmp_files:
            with open(tmp_csv, 'r') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    path = row[0]
                    score = row[1]
                    status = row[2]
                    scores_dict[path] = float(score) if status == "success" else -1
        
        # 写入最终结果（保持原始格式：path start end score）
        with open(output_csv, 'w') as f:
            for entry in original_entries:
                line = f"{entry['path']} {entry['start']} {entry['end']} {scores_dict.get(entry['path'], -1)}\n"
                f.write(line)
        return
    
    # 读取评分结果
    scores_dict = {}
    with open(output_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        for row in reader:
            path = row[0]
            score = row[1]
            status = row[2]
            scores_dict[path] = float(score) if status == "success" else -1

    # 写入最终结果（保持原始格式：path start end score）
    with open(output_csv, 'w') as f:
        for entry in original_entries:
            line = f"{entry['path']} {entry['start']} {entry['end']} {scores_dict.get(entry['path'], -1)}\n"
            f.write(line)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="segment_manifest.txt 路径")
    parser.add_argument("--output_csv", required=True, help="输出评分文件路径")
    parser.add_argument("--gpus", default="0", help="使用的GPU列表，如: 0,1")
    args = parser.parse_args()
    main(args.input_csv, args.output_csv, args.gpus)
