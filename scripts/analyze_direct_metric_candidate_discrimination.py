#!/usr/bin/env python
"""Read-only risk discrimination from saved candidate predictions.

NC's0.5 soft labels are excluded from the binary diagnostic, not training.
Within-scene AUC distinguishes candidate risk ordering from scene priors.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from run_direct_metric_visual_probe_pilot import CachedMetricProbeScenes
from calibrate_frozen_drivor_scorer import file_sha256, write_json
from run_metric_visual_evidence_pilot import verify_registration


def binary_risk_auc(targets, safe_logits):
    valid = np.isclose(targets, 0.) | np.isclose(targets, 1.)
    risk = np.isclose(targets[valid], 0.)
    positive_count = int(risk.sum())
    negative_count = len(risk) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None
    if np.ptp(safe_logits[valid]) <= 1e-5:
        return 0.5
    # Treat insignificantfloat32 weighted-sum noise as ties for constant inputs.
    ranks = rankdata(np.round(-safe_logits[valid], decimals=6), method="average")
    return float((ranks[risk].sum() - positive_count * (positive_count + 1) / 2)
                 / (positive_count * negative_count))


def main(configuration_path):
    configuration = json.loads(configuration_path.read_text())
    verify_registration(configuration)
    scene_cache = CachedMetricProbeScenes(configuration)
    rows = scene_cache.rows("locked_development")
    targets = scene_cache.targets[rows]
    saved = np.load(scene_cache.result / "evaluation_predictions.npz")
    measurements = {}
    for condition in ("unchanged", "shared", "metric_specific", "metric_specific_uniform_attention",
                      "metric_specific_swap_metric_attention", "metric_specific_patch_constant", "train_only_metric_prior"):
        logits = saved[condition + "_logits"]
        measurements[condition] = {}
        for metric_index, metric in enumerate(("NC", "DAC")):
            scene_auc = [binary_risk_auc(targets[index, :, metric_index], logits[index, :, metric_index])
                         for index in range(len(rows))]
            informative = [value for value in scene_auc if value is not None]
            measurements[condition][metric] = {
                "pooled_binary_risk_auc": binary_risk_auc(targets[:, :, metric_index].ravel(), logits[:, :, metric_index].ravel()),
                "mean_within_scene_binary_risk_auc": float(np.mean(informative)) if informative else None,
                "informative_scenes_with_both_binary_labels": len(informative),
                "excluded_partial_targets": int((~(np.isclose(targets[:, :, metric_index], 0.) | np.isclose(targets[:, :, metric_index], 1.))).sum()),
            }
    content = {"measurements": measurements, "script_sha256": file_sha256(Path(__file__)),
               "binary_target_policy": "Unsafe0 vs safe1; exclude0.5 NC from this diagnostic only. Risk prediction=-safe_logit. Treat flat logit ranges<=1e-5 as ties; otherwise round1e-6.",
               "interpretation": "Candidate risk ranking, not object/sky/road semantic localization. Pooled AUC can reflect scene differences; within-scene AUC uses only scenes with both classes.",
               "optimizer_steps": 0, "gpu_seconds": 0}
    write_json(scene_cache.result / "candidate_metric_discrimination.json", content)
    report_path = scene_cache.result / "report.md"
    report = report_path.read_text().split("\n## 후보별 위험 구분 추가 진단")[0]
    lines = ["", "## 후보별 위험 구분 추가 진단", "",
             "후보별 점수를 읽는 능력을 확인하기 위해 NC/DAC0(위험)와1(안전)만 이용해 AUC를 계산했다. NC0.5는 이 binary 진단에서만 제외했다. 장면내AUC는 두label이모두있는장면만집계하며semantic영역정확도는아니다.", "",
             "| 조건 | NC 장면내AUC | DAC 장면내AUC |", "|---|---:|---:|"]
    for condition in ("unchanged", "shared", "metric_specific", "metric_specific_patch_constant"):
        values = measurements[condition]
        lines.append(f"| {condition} | {values['NC']['mean_within_scene_binary_risk_auc']:.4f} | {values['DAC']['mean_within_scene_binary_risk_auc']:.4f} |")
    counts = measurements["unchanged"]
    lines.append(f"\n정보가있는장면수 NC{counts['NC']['informative_scenes_with_both_binary_labels']}/DAC{counts['DAC']['informative_scenes_with_both_binary_labels']}이다. BinaryNC에서제외한partialtarget은{counts['NC']['excluded_partial_targets']}개다. PooledAUC·개입별전체수치는candidate_metric_discrimination.json에저장했다. 재실행은동일환경에서 `python scripts/analyze_direct_metric_candidate_discrimination.py`.")
    report_path.write_text(report + "\n".join(lines) + "\n")
    print(json.dumps(measurements, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", type=Path, default=Path("configs/direct_metric_visual_probe/pilot_v1.json"))
    main(parser.parse_args().configuration)
