#!/usr/bin/env python3
"""
V3 模型权重下载脚本
==================
一键下载 V3 纯语义时间图异常检测系统所需的全部模型权重。

模型列表:
  1. Qwen2-VL-7B-Instruct  (~14 GB)  — 主 VLLM 感知模型
  2. Moondream2             (~3.6 GB) — 轻量级备选 VLLM
  3. all-MiniLM-L6-v2       (~80 MB)  — Sentence-BERT 语义 Re-ID

所有模型统一缓存到: <PROJECT_ROOT>/models/huggingface/

用法:
  conda activate eventvad_vllm
  python scripts/download_v3_models.py              # 下载全部
  python scripts/download_v3_models.py --only qwen  # 只下载 Qwen2-VL
  python scripts/download_v3_models.py --only moon   # 只下载 Moondream2
  python scripts/download_v3_models.py --only sbert  # 只下载 Sentence-BERT
  python scripts/download_v3_models.py --exclude moon # 排除 Moondream2
"""

import os
import sys
import argparse
import pathlib
import time

# ── 路径设置 ─────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]  # EventVAD/
MODELS_DIR = PROJECT_ROOT / "models"
HF_CACHE_DIR = MODELS_DIR / "huggingface"
SBERT_CACHE_DIR = HF_CACHE_DIR / "sbert"

# 设置 HuggingFace 缓存环境变量（必须在 import transformers 之前）
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")

# ── 模型定义 ─────────────────────────────────────────
MODELS = {
    "qwen": {
        "name": "Qwen2-VL-7B-Instruct",
        "repo_id": "Qwen/Qwen2-VL-7B-Instruct",
        "size": "~14 GB",
        "type": "transformers",
    },
    "moon": {
        "name": "Moondream2",
        "repo_id": "vikhyatk/moondream2",
        "size": "~3.6 GB",
        "type": "transformers",
    },
    "sbert": {
        "name": "all-MiniLM-L6-v2 (Sentence-BERT)",
        "repo_id": "sentence-transformers/all-MiniLM-L6-v2",
        "size": "~80 MB",
        "type": "sentence-transformers",
    },
}


def print_header():
    print("=" * 60)
    print("  EventVAD V3 — 模型权重下载工具")
    print("=" * 60)
    print(f"  项目根目录:   {PROJECT_ROOT}")
    print(f"  模型缓存目录: {HF_CACHE_DIR}")
    print(f"  SBERT 目录:   {SBERT_CACHE_DIR}")
    print("=" * 60)


def download_transformers_model(repo_id: str, model_name: str):
    """通过 huggingface_hub 下载 transformers 模型"""
    from huggingface_hub import snapshot_download

    print(f"\n{'─' * 50}")
    print(f"  正在下载: {model_name}")
    print(f"  仓库:     {repo_id}")
    print(f"  目标:     {HF_CACHE_DIR / 'hub'}")
    print(f"{'─' * 50}")

    t0 = time.time()
    local_path = snapshot_download(
        repo_id=repo_id,
        cache_dir=str(HF_CACHE_DIR / "hub"),
        resume_download=True,
    )
    elapsed = time.time() - t0

    print(f"  ✅ {model_name} 下载完成!")
    print(f"  📁 路径: {local_path}")
    print(f"  ⏱️  耗时: {elapsed:.1f}s")
    return local_path


def download_sbert_model(repo_id: str, model_name: str):
    """通过 sentence-transformers 下载 SBERT 模型"""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  ⚠️  sentence-transformers 未安装，尝试用 huggingface_hub 下载...")
        return download_transformers_model(repo_id, model_name)

    print(f"\n{'─' * 50}")
    print(f"  正在下载: {model_name}")
    print(f"  仓库:     {repo_id}")
    print(f"  目标:     {SBERT_CACHE_DIR}")
    print(f"{'─' * 50}")

    t0 = time.time()
    model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        cache_folder=str(SBERT_CACHE_DIR),
    )
    elapsed = time.time() - t0

    # 验证模型可用
    test_emb = model.encode(["test sentence"])
    assert test_emb.shape[1] > 0, "SBERT 模型加载验证失败"

    print(f"  ✅ {model_name} 下载完成!")
    print(f"  📁 路径: {SBERT_CACHE_DIR}")
    print(f"  📐 向量维度: {test_emb.shape[1]}")
    print(f"  ⏱️  耗时: {elapsed:.1f}s")

    del model
    return str(SBERT_CACHE_DIR)


def verify_transformers_model(repo_id: str, model_name: str):
    """验证 transformers 模型是否已缓存"""
    from huggingface_hub import try_to_load_from_cache, scan_cache_dir

    cache_info = scan_cache_dir(str(HF_CACHE_DIR / "hub"))
    for repo in cache_info.repos:
        if repo.repo_id == repo_id:
            size_gb = repo.size_on_disk / (1024 ** 3)
            print(f"  ✅ {model_name}: 已缓存 ({size_gb:.2f} GB)")
            return True
    print(f"  ❌ {model_name}: 未找到缓存")
    return False


def verify_sbert_model():
    """验证 SBERT 模型是否已缓存"""
    sbert_dir = SBERT_CACHE_DIR
    if sbert_dir.exists() and any(sbert_dir.iterdir()):
        total = sum(f.stat().st_size for f in sbert_dir.rglob("*") if f.is_file())
        size_mb = total / (1024 ** 2)
        print(f"  ✅ Sentence-BERT: 已缓存 ({size_mb:.1f} MB)")
        return True
    print(f"  ❌ Sentence-BERT: 未找到缓存")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="V3 模型权重下载工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/download_v3_models.py              # 下载全部
  python scripts/download_v3_models.py --only qwen  # 只下载 Qwen2-VL
  python scripts/download_v3_models.py --only sbert  # 只下载 SBERT
  python scripts/download_v3_models.py --exclude moon # 排除 Moondream2
  python scripts/download_v3_models.py --check       # 仅检查缓存状态
        """,
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(MODELS.keys()),
        help="只下载指定模型 (qwen, moon, sbert)",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        choices=list(MODELS.keys()),
        help="排除指定模型",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="仅检查模型缓存状态，不下载",
    )
    args = parser.parse_args()

    print_header()

    # 确定要处理的模型
    if args.only:
        target_keys = args.only
    else:
        target_keys = list(MODELS.keys())

    if args.exclude:
        target_keys = [k for k in target_keys if k not in args.exclude]

    if not target_keys:
        print("\n  ⚠️  没有需要处理的模型，退出。")
        return

    targets = {k: MODELS[k] for k in target_keys}

    # 显示计划
    print("\n📋 计划处理的模型:")
    for key, info in targets.items():
        print(f"   • {info['name']}  ({info['size']})")

    # 仅检查模式
    if args.check:
        print("\n🔍 检查缓存状态:")
        for key, info in targets.items():
            if info["type"] == "sentence-transformers":
                verify_sbert_model()
            else:
                try:
                    verify_transformers_model(info["repo_id"], info["name"])
                except Exception as e:
                    print(f"  ❌ {info['name']}: 检查失败 ({e})")
        return

    # 创建缓存目录
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    SBERT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 下载
    print(f"\n🚀 开始下载 ({len(targets)} 个模型)...\n")

    success = []
    failed = []
    t_total = time.time()

    for key, info in targets.items():
        try:
            if info["type"] == "sentence-transformers":
                download_sbert_model(info["repo_id"], info["name"])
            else:
                download_transformers_model(info["repo_id"], info["name"])
            success.append(info["name"])
        except KeyboardInterrupt:
            print("\n\n  ⚠️  用户中断，已下载的部分不会丢失（支持断点续传）。")
            sys.exit(1)
        except Exception as e:
            print(f"\n  ❌ {info['name']} 下载失败: {e}")
            failed.append((info["name"], str(e)))

    total_elapsed = time.time() - t_total

    # 汇总
    print(f"\n{'=' * 60}")
    print(f"  下载完成! 总耗时: {total_elapsed:.1f}s")
    print(f"{'=' * 60}")
    if success:
        print(f"  ✅ 成功: {', '.join(success)}")
    if failed:
        print(f"  ❌ 失败:")
        for name, err in failed:
            print(f"     • {name}: {err}")

    print(f"\n📁 模型缓存位置:")
    print(f"   HuggingFace: {HF_CACHE_DIR / 'hub'}")
    print(f"   SBERT:       {SBERT_CACHE_DIR}")
    print()


if __name__ == "__main__":
    main()
