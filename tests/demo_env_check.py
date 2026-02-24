"""
EventVAD 环境综合检查 demo
检查核心依赖是否可用 + torch/CUDA/CLIP/OpenCV 基本功能
"""
import sys
import time

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"

results = []


def check(name, fn):
    try:
        info = fn()
        print(f"  {PASS} {name}: {info}")
        results.append((name, True, info))
    except Exception as e:
        print(f"  {FAIL} {name}: {e}")
        results.append((name, False, str(e)))


def main():
    print("=" * 60)
    print("  EventVAD Conda 环境综合检查")
    print(f"  Python: {sys.version}")
    print("=" * 60)

    # ── 1. 基础依赖 ──
    print("\n[1/5] 基础依赖导入 ...")
    check("numpy", lambda: __import__("numpy").__version__)
    check("opencv", lambda: __import__("cv2").__version__)
    check("PIL/Pillow", lambda: __import__("PIL").__version__)
    check("einops", lambda: __import__("einops").__version__)
    check("decord", lambda: __import__("decord").__version__)

    # ── 2. PyTorch + CUDA ──
    print("\n[2/5] PyTorch + CUDA ...")
    import torch
    check("torch", lambda: torch.__version__)
    check("torchvision", lambda: __import__("torchvision").__version__)
    check("CUDA available", lambda: f"{torch.cuda.is_available()}")
    if torch.cuda.is_available():
        check("CUDA device", lambda: torch.cuda.get_device_name(0))
        check("CUDA version", lambda: torch.version.cuda)
        check("cuDNN version", lambda: str(torch.backends.cudnn.version()))

        def _gpu_tensor_test():
            a = torch.randn(256, 256, device="cuda")
            b = torch.randn(256, 256, device="cuda")
            c = a @ b
            torch.cuda.synchronize()
            return f"matmul OK, shape={tuple(c.shape)}, dtype={c.dtype}"
        check("GPU tensor 计算", _gpu_tensor_test)

    # ── 3. CLIP ──
    print("\n[3/5] CLIP 模型 ...")

    def _clip_test():
        import clip as clip_mod
        avail = clip_mod.available_models()
        return f"available models: {avail[:3]}..."
    check("clip (openai)", _clip_test)

    def _transformers_clip():
        from transformers import CLIPModel, CLIPProcessor
        return "CLIPModel & CLIPProcessor importable"
    check("transformers CLIP", _transformers_clip)

    # ── 4. Ultralytics (YOLO) ──
    print("\n[4/5] Ultralytics YOLO ...")
    check("ultralytics", lambda: __import__("ultralytics").__version__)

    # ── 5. 项目内模块导入 ──
    print("\n[5/5] EventVAD 项目模块 ...")
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

    def _import_v5_config():
        from v5.config import SystemConfig
        return f"device={SystemConfig.device}, version={SystemConfig.version}"
    check("v5.config", _import_v5_config)

    def _import_graph():
        from v5.graph.graph_builder import GraphBuilder
        from v5.graph.narrative_generator import NarrativeGenerator
        from v5.graph.structures import EntityGraph
        return "GraphBuilder, NarrativeGenerator, EntityGraph OK"
    check("v5.graph", _import_graph)

    def _import_tracking():
        from v5.tracking.entity_tracker import EntityTracker
        from v5.tracking.motion_extractor import MotionExtractor
        return "EntityTracker, MotionExtractor OK"
    check("v5.tracking", _import_tracking)

    def _import_semantic():
        from v5.semantic.node_trigger import NodeTrigger
        return "NodeTrigger OK"
    check("v5.semantic", _import_semantic)

    # ── 汇总 ──
    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    failed = [name for name, ok, _ in results if not ok]
    if failed:
        print(f"  结果: {passed}/{total} 通过, 失败项: {failed}")
    else:
        print(f"  结果: {passed}/{total} 全部通过 ✓")
    print("=" * 60)
    return len(failed) == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
