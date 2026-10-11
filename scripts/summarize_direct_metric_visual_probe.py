#!/usr/bin/env python
"""Package interpretation and read-only tuning-loss curves; no new selection."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as functional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from run_direct_metric_visual_probe_pilot import CachedMetricProbeScenes, load_probe, predict
from run_metric_visual_evidence_pilot import verify_registration
from calibrate_frozen_drivor_scorer import file_sha256, protected_training_audit, recording_bootstrap, write_json


def main(configuration_path):
    configuration = json.loads(configuration_path.read_text())
    registration = verify_registration(configuration)
    torch.set_num_threads(configuration["cpu_threads"])
    scene_cache = CachedMetricProbeScenes(configuration)
    results = json.loads((scene_cache.result / "results.json").read_text())
    diagnostics = json.loads((scene_cache.result / "functional_diagnostics.json").read_text())
    discrimination = json.loads((scene_cache.result / "candidate_metric_discrimination.json").read_text())
    started = time.perf_counter()
    tuning_rows = scene_cache.rows("tuning")
    tuning_curves = {}
    for condition in ("shared", "metric_specific"):
        tuning_curves[condition] = []
        native_training = results["training"]["conditions"][condition]
        for step in configuration["checkpoint_steps"]:
            model, _ = load_probe(configuration, condition, torch.device("cpu"), step=step)
            model.requires_grad_(False)
            _, logits = predict(configuration, scene_cache, model, tuning_rows, torch.device("cpu"))
            component_bce = functional.binary_cross_entropy_with_logits(
                torch.from_numpy(logits[..., :2]), torch.from_numpy(scene_cache.targets[tuning_rows, :, :2])).item()
            original_tuning_pdms = next(record["tuning_pdms"] for record in native_training["logs"] if record["step"] == step)
            tuning_curves[condition].append({"step": step, "NC_DAC_BCE_cpu": component_bce,
                                             "previously_recorded_tuning_PDMS_gpu": original_tuning_pdms})
    curves = {"curves": tuning_curves, "cpu_wall_seconds": time.perf_counter() - started,
              "optimizer_steps": 0, "gpu_seconds": 0., "encoder_calls": 0,
              "selection_unchanged": results["selected_steps"],
              "interpretation": "Read-only loss diagnosis on already registered tuning64/checkpoints. Does not reselect a primary model from development scores."}
    write_json(scene_cache.result / "tuning_metric_loss_curves.json", curves)
    dev_rows = scene_cache.rows("locked_development")
    labels = scene_cache.targets[dev_rows]
    predictions = np.load(scene_cache.result / "evaluation_predictions.npz")
    groups = [scene_cache.records[row]["recording_group"] for row in dev_rows]
    chosen = {name: predictions[name + "_scores"].argmax(1) for name in ("unchanged", "shared", "metric_specific")}
    difference = (labels[np.arange(len(dev_rows)), chosen["metric_specific"], 6]
                  - labels[np.arange(len(dev_rows)), chosen["unchanged"], 6]) * 100
    versus_original_interval = recording_bootstrap(difference, groups, seed=0, repetitions=2000)
    upper = np.array([record["current_upper6rows_density_NC_DAC"] for record in diagnostics["rows"]]).mean(0)
    lower = np.array([record["current_lower6rows_density_NC_DAC"] for record in diagnostics["rows"]]).mean(0)
    summary = {
        "specific_minus_original_pdms": float(difference.mean()),
        "specific_minus_original_recording_95ci": versus_original_interval,
        "shared_specific_all_selected_indices_identical": bool(np.array_equal(chosen["shared"], chosen["metric_specific"])),
        "fixed12_upper6rows_attention_density_NC_DAC": upper.tolist(),
        "fixed12_lower6rows_attention_density_NC_DAC": lower.tolist(),
        "semantic_sky_road_masks_used": False,
        "original_source_registration": registration["sources"],
        "postprocessor_sha256": file_sha256(Path(__file__)),
        "candidate_discrimination_postprocessor_sha256": file_sha256(PROJECT_ROOT / "scripts/analyze_direct_metric_candidate_discrimination.py"),
        "checkpoint_sha256": {path.name: file_sha256(path) for path in scene_cache.output.glob("*_step*.pt")},
        "protected_training": protected_training_audit(configuration),
    }
    write_json(scene_cache.result / "interpretation_and_provenance.json", summary)
    report_path = scene_cache.result / "report.md"
    report = report_path.read_text()
    title, body = report.split("\n", 1)
    measurement = discrimination["measurements"]
    original, specific = results["summaries"]["unchanged"], results["summaries"]["metric_specific"]
    interpretation = f"""
## 결과 해석

**시각 입력을 실제로 읽는 경로는 만들었으나, NC·DAC별로 읽는 영역을 분리하는 추가 이득은 확인하지 못했다.**

- 기존 {original['selected_pdms']:.2f} → 새 shared/specific {specific['selected_pdms']:.2f} PDMS다. Specific−기존은 {difference.mean():+.2f}점이며 recording bootstrap 95% 구간은 [{versus_original_interval[0]:+.2f}, {versus_original_interval[1]:+.2f}]이다. 단일 seed/기존 개발셋의 결과이므로 성능 향상을 확정하지 않는다.
- Shared와 Specific은 **192장면 모두 같은 후보**를 선택했다. NC/DAC attention 교환도 선택을 바꾸지 않았다. 지표별 spatial reading의 planning 이득은 입증되지 않았다.
- NC 실패는 기존 3/192 → 새 모델 5/192, DAC 실패는 21/192로 같았다. 사전 등록한 안전 기준을 충족하지 못했다. 평균 PDMS 증가만으로 성공으로 판단하지 않는다.
- NC/DAC 평균 BCE는 기존 {original['mean_NC_DAC_BCE']:.4f}, Specific {specific['mean_NC_DAC_BCE']:.4f}로 나빠졌다. 장면 안에서 후보 위험을 구분하는 AUC도 기존 NC {measurement['unchanged']['NC']['mean_within_scene_binary_risk_auc']:.3f}/DAC {measurement['unchanged']['DAC']['mean_within_scene_binary_risk_auc']:.3f} 대비 새 NC {measurement['metric_specific']['NC']['mean_within_scene_binary_risk_auc']:.3f}/DAC {measurement['metric_specific']['DAC']['mean_within_scene_binary_risk_auc']:.3f}였다. 이는 원 scorer보다 해당 지표를 더 잘 이해했다는 근거가 없다.
- 다만 train-only 상수 prior보다 BCE가 작고, visual feature를 상수로 바꾸면 BCE가 커지며 선택이 15/192 바뀌었다. 사전 등록한 약한 시각 읽기 진단은 통과했다. 이것이 semantic 특화나 인과적 정보 가치의 증명은 아니다.
- 같은 이미지·후보·공통 색상 척도에서 이전의 하늘 쪽 강조는 줄고 도로가 있는 하단 쪽으로 가중치가 옮겨지는 모습이 나왔다. 고정12장면의 상단/하단 농도는 NC {upper[0]:.3f}/{lower[0]:.3f}, DAC {upper[1]:.3f}/{lower[1]:.3f}다. 좌표 기준 측정이며 하늘/도로 segmentation을 평가한 것이 아니다. **NC는 차량, DAC는 차선이라는 분리는 아직 확인하지 못했다.**
- 마지막 500-step의 개발 PDMS는 shared {results['last_step500_pdms_without_dev_selection']['shared']:.2f}, specific {results['last_step500_pdms_without_dev_selection']['metric_specific']:.2f}였다. 체크포인트 선택은 튜닝64에서 결정한100step을 유지했다. 추가 epoch 확대 근거로 삼지 않는다.

### 다음 한 실험 — 아직 미실행

**BCE만 사용하는 NC/DAC 직접 예측 대조군**을 같은 예산에서 비교하는 것을 추천한다. 현재 composite PDMS ranking loss는 고정된 다른 네 지표의 예측 오차를 NC/DAC 출력으로 보상하게 만들 가능성이 있다. 이 가능성은 실패 원인으로 확정하지 않았다. 지표별 표현의 학습 목적을 먼저 분리하고, 해당 비교의 checkpoint 선택 기준도 실행 전에 등록해야 한다. 추가 본학습/semantic 감독은 자동 시작하지 않았다.

학습 step 시간 합계는 미니배치 로딩·동기화를 포함한 파이프라인 경과시간이다. 순수 GPU kernel 시간으로 해석하지 않는다. 짧은 profiling의 약16.8초 예상과 실제 약378.4초가 달랐으며, 긴 step의 세부 I/O/대기 원인을 계측하지 않았다. 실제 비용을 기준으로 보고한다.
"""
    if "\n## 결과 해석\n" not in body:
        report = title + "\n" + interpretation + body
    # CPU-only readout of tuning losses preserves the originally selected model.
    report = report.split("\n## 튜닝 loss 추가 진단")[0]
    lines = ["", "## 튜닝 loss 추가 진단", "",
             "저장된100/250/500step 모델을 CPU에서 튜닝64의NC/DAC BCE로만 진단했다. 아래PDMS는 본실행에서 기록한GPU값이며 재선택하지 않았다.", "",
             "| 조건 | step | 튜닝 NC/DAC BCE | 기존 튜닝 PDMS |", "|---|---:|---:|---:|"]
    for condition, checkpoints in tuning_curves.items():
        for checkpoint in checkpoints:
            lines.append(f"| {condition} | {checkpoint['step']} | {checkpoint['NC_DAC_BCE_cpu']:.5f} | {checkpoint['previously_recorded_tuning_PDMS_gpu']:.2f} |")
    lines.extend(["", "추가 읽기 전용 진단 재실행:", "", "```bash",
                  "CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src runtime/environments/kjs-nc-dac-visual-probe/bin/python scripts/analyze_direct_metric_candidate_discrimination.py",
                  "CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src runtime/environments/kjs-nc-dac-visual-probe/bin/python scripts/summarize_direct_metric_visual_probe.py",
                  "```"])
    report_path.write_text(report + "\n".join(lines) + "\n")
    print(json.dumps({"tuning_curves": tuning_curves, "versus_original_ci": versus_original_interval,
                      "upper_density": upper.tolist(), "lower_density": lower.tolist()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", type=Path, default=Path("configs/direct_metric_visual_probe/pilot_v1.json"))
    main(parser.parse_args().configuration)
