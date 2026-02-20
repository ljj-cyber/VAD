"""
Abuse030 零检出诊断脚本 —— 逐帧分析 motion + YOLO 的原始输出，
找出 0 entity 的根因。

conda run -n eventvad python -m v5.diag_abuse030
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

from v5.config import MotionConfig, YoloDetectorConfig, HybridDetectorConfig
from v5.tracking.motion_extractor import MotionExtractor
from v5.tracking.yolo_detector import YoloDetector

VIDEO = "/data/liuzhe/EventVAD/src/event_seg/videos/ucf_crime/Anomaly-Videos-Part-1/Abuse/Abuse030_x264.mp4"
GT_INTERVAL = (1275, 1360)
SAMPLE_EVERY = 2


def main():
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {VIDEO}")
    print(f"Frames: {total_frames}, FPS: {fps:.1f}, Duration: {total_frames/fps:.1f}s")
    print(f"GT anomaly: frames {GT_INTERVAL[0]}-{GT_INTERVAL[1]} "
          f"({GT_INTERVAL[0]/fps:.1f}s - {GT_INTERVAL[1]/fps:.1f}s)")
    print()

    # --- 独立初始化两个检测器 ---
    motion_ext = MotionExtractor(MotionConfig())
    yolo_det = YoloDetector(YoloDetectorConfig(), lazy_load=True)

    frame_idx = 0
    processed = 0

    # 统计
    motion_energies = []
    motion_region_counts = []
    yolo_raw_counts = []
    yolo_detections_all = []
    frames_with_motion = 0
    frames_with_yolo = 0

    # 对 GT 区间附近的帧做详细分析
    gt_start, gt_end = GT_INTERVAL
    detail_start = max(0, gt_start - 100)
    detail_end = min(total_frames, gt_end + 100)

    print("=" * 70)
    print("Phase 1: Full video motion energy scan")
    print("=" * 70)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % SAMPLE_EVERY != 0:
            frame_idx += 1
            continue

        # --- Motion ---
        regions = motion_ext.extract(frame)
        energy = motion_ext.compute_frame_energy()
        motion_energies.append((frame_idx, energy))
        motion_region_counts.append((frame_idx, len(regions)))
        if regions:
            frames_with_motion += 1

        # --- YOLO: 只在 GT 附近区间跑 (省时间) ---
        if detail_start <= frame_idx <= detail_end:
            yolo_regions = yolo_det.detect(frame, force=True)
            yolo_raw_counts.append((frame_idx, len(yolo_regions)))
            if yolo_regions:
                frames_with_yolo += 1
            for yr in yolo_regions:
                yolo_detections_all.append({
                    "frame": frame_idx,
                    "class": yr.class_name,
                    "conf": yr.confidence,
                    "box": (yr.x, yr.y, yr.w, yr.h),
                    "area": yr.area,
                })

        processed += 1
        frame_idx += 1

    cap.release()

    # --- 报告 ---
    print(f"\nProcessed {processed} frames (sample_every={SAMPLE_EVERY})")
    print(f"Frames with motion regions: {frames_with_motion}/{processed} "
          f"({frames_with_motion/max(processed,1)*100:.1f}%)")

    # 动能统计
    energies = [e for _, e in motion_energies]
    print(f"\nMotion energy stats:")
    print(f"  mean={np.mean(energies):.6f}, std={np.std(energies):.6f}")
    print(f"  max={np.max(energies):.6f}, p95={np.percentile(energies, 95):.6f}")
    print(f"  p99={np.percentile(energies, 99):.6f}")

    # GT 区间附近的动能
    gt_energies = [(fi, e) for fi, e in motion_energies
                   if detail_start <= fi <= detail_end]
    gt_only_energies = [(fi, e) for fi, e in motion_energies
                        if gt_start <= fi <= gt_end]
    if gt_energies:
        print(f"\nMotion energy near GT ({detail_start}-{detail_end}):")
        print(f"  mean={np.mean([e for _,e in gt_energies]):.6f}")
        print(f"  max={np.max([e for _,e in gt_energies]):.6f}")
    if gt_only_energies:
        print(f"\nMotion energy IN GT ({gt_start}-{gt_end}):")
        print(f"  mean={np.mean([e for _,e in gt_only_energies]):.6f}")
        print(f"  max={np.max([e for _,e in gt_only_energies]):.6f}")

    # GT 区间的帧差区域检出情况
    gt_motion_regions = [(fi, n) for fi, n in motion_region_counts
                         if gt_start <= fi <= gt_end]
    gt_with_regions = sum(1 for _, n in gt_motion_regions if n > 0)
    print(f"\nMotion regions IN GT interval:")
    print(f"  Frames with regions: {gt_with_regions}/{len(gt_motion_regions)}")

    # 列出 GT 附近所有有动能的帧
    print(f"\n--- Frames with motion energy > 0.005 near GT ---")
    high_energy_frames = [(fi, e) for fi, e in motion_energies
                          if e > 0.005 and detail_start <= fi <= detail_end]
    if high_energy_frames:
        for fi, e in high_energy_frames[:30]:
            in_gt = "★GT" if gt_start <= fi <= gt_end else "   "
            regions_at = [n for f2, n in motion_region_counts if f2 == fi]
            n_reg = regions_at[0] if regions_at else 0
            print(f"  frame={fi:5d} ({fi/fps:6.1f}s) energy={e:.6f} regions={n_reg} {in_gt}")
        if len(high_energy_frames) > 30:
            print(f"  ... and {len(high_energy_frames)-30} more")
    else:
        print("  NONE!")

    # YOLO 结果
    print(f"\n{'='*70}")
    print(f"Phase 2: YOLO detections near GT ({detail_start}-{detail_end})")
    print(f"{'='*70}")
    print(f"YOLO scanned frames: {len(yolo_raw_counts)}")
    print(f"Frames with YOLO detections: {frames_with_yolo}")

    if yolo_detections_all:
        # 按类别汇总
        cls_counts = defaultdict(int)
        cls_confs = defaultdict(list)
        for d in yolo_detections_all:
            cls_counts[d["class"]] += 1
            cls_confs[d["class"]].append(d["conf"])

        print(f"\nYOLO detection summary (near GT):")
        for cls, cnt in sorted(cls_counts.items(), key=lambda x: -x[1]):
            confs = cls_confs[cls]
            print(f"  {cls}: {cnt} detections, "
                  f"conf mean={np.mean(confs):.3f} "
                  f"min={np.min(confs):.3f} max={np.max(confs):.3f}")

        # 列出 GT 区间内的 YOLO 检测
        gt_yolo = [d for d in yolo_detections_all
                    if gt_start <= d["frame"] <= gt_end]
        print(f"\nYOLO detections IN GT interval ({gt_start}-{gt_end}):")
        if gt_yolo:
            for d in gt_yolo:
                print(f"  frame={d['frame']} class={d['class']} "
                      f"conf={d['conf']:.3f} box={d['box']} area={d['area']}")
        else:
            print("  NONE!")

        # 列出 GT 附近所有 person 检测
        person_dets = [d for d in yolo_detections_all if d["class"] == "person"]
        print(f"\nAll 'person' detections near GT:")
        if person_dets:
            for d in person_dets[:30]:
                in_gt = "★GT" if gt_start <= d["frame"] <= gt_end else "   "
                print(f"  frame={d['frame']} conf={d['conf']:.3f} "
                      f"box={d['box']} area={d['area']} {in_gt}")
        else:
            print("  NONE!")
    else:
        print("\nNo YOLO detections at all!")

    # --- 抽几帧保存截图看看视频内容 ---
    print(f"\n{'='*70}")
    print(f"Phase 3: Saving sample frames for visual inspection")
    print(f"{'='*70}")
    sample_frames = [
        gt_start - 50, gt_start, gt_start + 20,
        (gt_start + gt_end) // 2, gt_end, gt_end + 50
    ]
    out_dir = Path("/data/liuzhe/EventVAD/output/v5/bad_case_test/diag_abuse030")
    out_dir.mkdir(parents=True, exist_ok=True)

    cap2 = cv2.VideoCapture(VIDEO)
    for target_frame in sample_frames:
        target_frame = max(0, min(target_frame, total_frames - 1))
        cap2.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ok, frame = cap2.read()
        if ok:
            out_path = out_dir / f"frame_{target_frame:05d}.jpg"
            cv2.imwrite(str(out_path), frame)
            print(f"  Saved: {out_path} (frame {target_frame}, {target_frame/fps:.1f}s)")
    cap2.release()

    # --- 额外: 全视频 YOLO 在 low conf 下能检出什么 ---
    print(f"\n{'='*70}")
    print(f"Phase 4: YOLO with very low threshold on GT frames")
    print(f"{'='*70}")
    from v5.config import YoloDetectorConfig as YC
    low_cfg = YC()
    low_cfg.confidence_threshold = 0.05  # 极低阈值
    low_cfg.max_det = 20
    yolo_low = YoloDetector(low_cfg, lazy_load=False)

    cap3 = cv2.VideoCapture(VIDEO)
    sample_detail_frames = list(range(gt_start, gt_end + 1, 10))
    for target_frame in sample_detail_frames:
        cap3.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ok, frame = cap3.read()
        if not ok:
            continue
        yolo_low.reset()
        low_regions = yolo_low.detect(frame, force=True)
        if low_regions:
            for yr in low_regions:
                print(f"  frame={target_frame} class={yr.class_name} "
                      f"conf={yr.confidence:.3f} box=({yr.x},{yr.y},{yr.w},{yr.h}) "
                      f"area={yr.area}")
        else:
            print(f"  frame={target_frame}: NO detections even at conf=0.05")
    cap3.release()

    # --- HybridDetector 融合阈值分析 ---
    print(f"\n{'='*70}")
    print(f"Root cause analysis")
    print(f"{'='*70}")
    cfg = MotionConfig()
    hcfg = HybridDetectorConfig()
    print(f"Motion min_region_area: {cfg.min_region_area}")
    print(f"Motion min_crop_size: {cfg.min_crop_size}")
    print(f"Motion diff_threshold: {cfg.diff_threshold} (adaptive min: {cfg.adaptive_threshold_min})")
    print(f"Hybrid yolo_only_min_conf: {hcfg.yolo_only_min_conf}")
    print(f"Hybrid motion_only_min_energy: {hcfg.motion_only_min_energy}")
    print(f"YOLO confidence_threshold: {YoloDetectorConfig().confidence_threshold}")

    max_e = max(energies) if energies else 0
    if max_e < 0.01:
        print(f"\n★ DIAGNOSIS: Video has extremely low motion (max energy={max_e:.6f})")
        print(f"  → Frame differencing cannot detect any significant motion regions")
        if not yolo_detections_all:
            print(f"  → YOLO also failed to detect objects near GT interval")
            print(f"  → Likely cause: very small/distant subjects, or non-visual abuse (verbal)")
        else:
            print(f"  → YOLO detected some objects but they were filtered by thresholds")
    else:
        if not yolo_detections_all and frames_with_motion == 0:
            print(f"\n★ DIAGNOSIS: Both detectors failed completely")
        elif frames_with_motion > 0 and frames_with_yolo == 0:
            print(f"\n★ DIAGNOSIS: Motion detected but YOLO missed everything")


if __name__ == "__main__":
    main()
