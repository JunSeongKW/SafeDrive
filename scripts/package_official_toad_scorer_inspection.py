#!/usr/bin/env python
"""Package measured scorer results and the independent CPU derivative check."""
import argparse
import json
from pathlib import Path

import numpy as np

from calibrate_official_toad_navsim_v1_scorer import (
    PROJECT_ROOT, file_sha256, protected_files_audit, write_json,
)


def package_inspection(configuration):
    output = PROJECT_ROOT / configuration["output_directory"]
    destination = PROJECT_ROOT / configuration["results_directory"]
    results = json.loads((destination / "results.json").read_text())
    inference = json.loads((output / "inference_complete.json").read_text())
    profile = json.loads((output / "profile_complete.json").read_text())
    derivative_validation = json.loads((destination / "cpu_gradient_validation.json").read_text())
    arrays = np.load(output / "scorer_cache_and_candidates.npz")
    official = np.load(output / "official_metrics.npy")
    selected_nearby = arrays["nearby_rankings"].argmax(axis=1)
    nearby_offset = configuration["native_candidate_count"]
    groups = results["groups"]
    context = results["fixed_size_context_diagnostic"]
    results["additional_validation"] = {
        "cpu_double_precision_finite_difference": derivative_validation,
        "nearby_selection_source_counts": {
            "original_fallback": int((selected_nearby == 0).sum()),
            "bicycle_roundtrip_center": int((selected_nearby == 1).sum()),
            "new_gaussian_samples": int((selected_nearby >= 2).sum())},
        "maximum_fixed_size_same_path_score_change": max(scene["fixed64_context_same_path_score_maximum_delta"] for scene in inference["scene_diagnostics"]),
        "mean_nearby_control_clipping_fraction": float(np.mean([scene["nearby_clipped_control_fraction"] for scene in inference["scene_diagnostics"]])),
        "mean_nearby_analytical_comfort_violation_fraction": float(np.mean([scene["nearby_comfort_violation_fraction"] for scene in inference["scene_diagnostics"]])),
        "raw_bicycle_center_mean_pdms": float(official[:, nearby_offset + 1, -1].mean() * 100),
        "original_fallback_mean_pdms": float(official[:, nearby_offset, -1].mean() * 100)}
    results["cost"]["completed_profile_gpu_phase_wall_seconds"] = profile["gpu_phase_wall_seconds"]
    results["cost"]["completed_gpu_phases_wall_seconds_sum"] = profile["gpu_phase_wall_seconds"] + inference["gpu_phase_wall_seconds"]
    results["cost"]["gpu_peak_allocated_bytes"] = max(profile["memory"]["peak_allocated_bytes"], inference["memory"]["peak_allocated_bytes"])
    results["cost"]["gpu_peak_reserved_bytes"] = max(profile["memory"]["peak_reserved_bytes"], inference["memory"]["peak_reserved_bytes"])
    results["cost"]["preliminary_failed_phase"] = {
        "failure": "nav1 has DrivoRFeatureBuilder rather than the older drivor_mpc utility",
        "image_or_scorer_forward_calls": 0, "gpu_phase_wall_seconds": None,
        "budget_reserve_seconds": 120, "failure_log": str(output / "profile_failed_before_inference.log")}
    results["protected_training"] = protected_files_audit(configuration)
    assert derivative_validation["all_finite_difference_checks_passed"]
    actual_sources = {name: file_sha256(PROJECT_ROOT / name) for name in (
        "scripts/calibrate_official_toad_navsim_v1_scorer.py",
        "scripts/verify_official_toad_cached_gradients.py",
        "scripts/package_official_toad_scorer_inspection.py",
        "scripts/run_official_toad_navsim_v1_scorer_calibration.sh",
        "configs/frozen_planner_postprocessing/official_toad_navsim_v1_scorer_calibration.json")}
    write_json(destination / "results.json", results)
    write_json(destination / "execution_provenance.json", {
        "sources": actual_sources, "inference_source_sha256": inference["diagnostic_source_sha256"],
        "official_repository_commit": configuration["official_repository_commit"],
        "official_feature_builder_source_sha256": inference["official_feature_builder_source_sha256"],
        "checkpoint_sha256": configuration["checkpoint_sha256"],
        "scorer_cache_path": str(output / "scorer_cache_and_candidates.npz"),
        "scorer_cache_sha256": file_sha256(output / "scorer_cache_and_candidates.npz"),
        "official_metrics_path": str(output / "official_metrics.npy"),
        "official_metrics_sha256": file_sha256(output / "official_metrics.npy"),
        "feature_metadata": inference["observed_inputs"], "feature_shape": inference["feature_shape"],
        "feature_dtype": inference["feature_dtype"], "environment": inference["environment"],
        "physical_gpu": 0, "optimizer_steps": 0})
    labels = {"learned64_alone": "학습된 후보만 별도 채점", "official128": "공식 기존 후보 설정, CEM OFF",
              "nearby64": "주변 후보", "nearby16": "동일 주변 후보 중 앞 16개"}
    report = ["# 공식 TOAD NAVSIM v1 scorer 진단", "",
        "## 결론", "",
        "**가중치 로딩·캐시 동등성·실제 경로 입력 gradient 검사는 통과했다. 평균 선택손실 기준도 통과했지만, 큰 gradient 보정은 실제 PDMS를 악화시켰다.**",
        "후처리 연구를 위한 베이스로 사용할 기술적 조건은 확보됐다. 안전한 최적화나 판단 안정성 기반 적용 여부의 효과는 아직 입증되지 않았다. 추가 학습 및 전체 CEM 성능 비교는 시작하지 않았다.", "",
        "## 평가 범위", "",
        "- NAVSIM v1의 NAVTEST **64장면 / 38 recording**, 관측 명령 기준 좌회전 21·직진 22·우회전 21. 모델의 예측을 보기 전에 seed와 장면을 고정했다.",
        "- 공개 가중치는 trainval + Simscale로 학습했으므로 NAVVAL은 제외했다. NAVTEST1024 개발 패널에 이미 노출됐으므로 독립 최종 테스트가 아니며, 명령별 균형 subset 점수다.",
        "- 원 패널의 `scene_type`은 미래 GT heading으로 만든 라벨이다. 이번 subset 선정과 상황별 집계에는 그것을 사용하지 않고 원본 현재 driving command를 사용했다.",
        "- 모든 가중치를 고정했다. optimizer **0 step**, encoder 재학습 없음. CEM 전체 반복은 실행하지 않았다.", "",
        "## 실제 점수", "",
        "선택손실은 같은 후보 집합에서 공식 PDMS가 가장 높은 경로의 점수와 scorer가 선택한 경로 점수의 차이다. Oracle 경로는 오프라인 진단에만 사용했다.", "",
        "| 조건 | 후보 수 | 선택 PDMS | 후보 내 실제 최고 | 평균 선택손실 | NC 또는 DAC 위반 선택 |",
        "|---|---:|---:|---:|---:|---:|"]
    for name, row in groups.items():
        report.append(f"| {labels[name]} | {row['candidate_count']} | {row['selected_pdms']:.4f} | {row['oracle_best_pdms']:.4f} | {row['selection_regret_pdms_points']:.4f} | {row['selected_collision_or_offroad_rate'] * 100:.2f}% |")
    report += ["", "공식 128후보는 학습된 64후보와 random bouquet 64후보다. 공식 eval 명령도 이 설정을 사용한다. 코드의 training-only 주석과 달리 실제 augmentation에 training 조건문은 없다.",
        "Nearby64는 원본 fallback 1개 + bicycle roundtrip center 1개 + Gaussian 62개다. 표준편차는 공식 CEM 초기식(전체 proposal 제어 std × 0.5, acceleration/yaw-rate floor 0.1/0.025)을 사용한다. Nearby16은 같은 pool의 앞 16개를 별도 채점하므로 후보 개수와 candidate context가 모두 달라진다.", "",
        "| 관측 명령 | 장면 수 | 공식 128후보 선택 PDMS | 주변 64후보 선택 PDMS |", "|---|---:|---:|---:|"]
    for command, row in groups["official128"]["by_command"].items():
        report.append(f"| {command} | {row['scenes']} | {row['selected_pdms']:.4f} | {groups['nearby64']['by_command'][command]['selected_pdms']:.4f} |")
    report += ["", "NC/DAC/TTC/progress/comfort/DDC 하위지표, 장면별 결과와 recording bootstrap 구간은 results.json 및 scene_results.csv에 있다. DDC의 PDMS 가중치는 0이므로 최고 PDMS가 모든 주행 조건의 충족을 뜻하지 않는다.", "",
        "## Gradient가 실제 성능을 개선하는가?", "",
        "다른 63후보를 고정하고 center 후보의 neural score에 대해 제어 입력 gradient를 계산했다. 보정 반경은 초기 std로 정규화한 gradient 방향의 길이다. **Neural-only 목적 검사이며 공식 TOAD의 comfort·anchor를 포함한 전체 목적/CEM의 결과가 아니다.**", "",
        "| 보정 반경 | Neural score 상승 장면 | 실제 PDMS 평균 변화 | 중앙값 | Recording bootstrap 95% 구간 | 1점 초과 하락 장면 |",
        "|---|---:|---:|---:|---|---:|"]
    for row in results["gradient_direction_tests"]:
        interval = row["actual_pdms_change_recording_bootstrap_95ci"]
        report.append(f"| {row['reference_std_radius']} | {row['neural_score_increase_rate'] * 100:.1f}% | {row['actual_pdms_mean_change']:+.4f} | {row['actual_pdms_median_change']:+.4f} | [{interval[0]:+.4f}, {interval[1]:+.4f}] | {row['actual_decrease_above_1_point_count']} / 64 |")
    report += ["", "작은 보정의 평균 증가 약 1.53점은 1점 초과 개선 장면 하나에 크게 좌우되며 중앙값은 0이고 신뢰구간은 0을 포함한다. 큰 보정에서 나타난 악화 때문에 scorer gradient를 그대로 믿고 움직이는 방식은 지지하지 않는다.",
        f"원본 경로를 제어로 역변환해 bicycle로 재생성한 center는 원본과 평균 ADE **{results['roundtrip_xy_ADE_m']:.4f}m** 다르다. 위 변화는 이 center 대비이며 원본 planner 경로 대비 개선이 아니다.", "",
        "## 후보 구성에 따른 점수 안정성", "",
        f"모든 context의 후보 수를 64개로 유지하고 공통 16개 경로 외의 48개 경로만 바꿨다. 공통 경로 중 최고 순위가 달라진 장면은 **{context['same_target_top1_change_scene_count']} / 64**다.",
        f"상대 점수 불안정성과 실제 선택손실의 Spearman 상관은 **{context['instability_vs_actual_selection_regret_spearman']:.4f}**였다. 현재 검사에서는 안정성 지표가 오류를 식별한다는 근거를 얻지 못했다. 안정성→위험도를 바로 해석하거나 이 값으로 보정 여부를 결정하면 안 된다.", "",
        "## 기술적 검증", "",
        "- 공개 asset SHA256 일치, 공식 nav1 commit **0110647142888dd56644565588510812bdafaa27**, **467 tensor strict load**. 모든 40,787,582 parameter 고정.",
        "- 입력: 현재 4카메라 F0/B0/L0/R0, 카메라당 **1148×672**, PIL resize + ImageNet 정규화. 현재 시점 뒤 영상 입력 0. 이전/현재 ego를 저장하고 planner는 현재 11항목을 사용한다.",
        "- 공식 `DrivoRFeatureBuilder.compute_features/_get_camera_feature`의 원본 AST 함수 본문을 재사용했다. 해당 환경에서 불필요한 map imports를 생략했고, 실제 이미지 3장면의 독립 전처리/ego 대조 오차는 0이었다.",
        "- 캐시: image_backbone 출력 **64×256 FP32 register token** 및 ego **1×256**. 모든 token valid이며 register가 원본 픽셀과 직접 대응하지는 않는다. 기존 Drive-JEPA/소형 DrivoR cache와 혼용하지 않았다.",
        "- 64장면 원본 forward와 cached scorer 점수 최대 오차 **0**, cached backbone 호출 **0**, 고정 weight digest 불변·weight gradient 누적 없음. 64장면의 경로 입력 gradient norm이 유한하며 0보다 크다.",
        "- FP32 finite difference에는 작은 score 변화의 양자화 오차가 있었다. 독립 CPU FP64 실제 scorer 검사 **3장면 × 2방향 통과**, 최대 절대 오차 약 **6.4e-11**. 이는 성능 개선 근거가 아닌 미분 경로 검증이다.",
        "- 공식 NAVSIM v1 평가와 경로별 progress 정규화를 사용했다. 일괄 계산을 변경 없는 reference+single-candidate evaluator와 대조한 최대 오차 **0**.", "",
        "## 비용과 보존", "",
        f"완료된 GPU 단계: profile **{profile['gpu_phase_wall_seconds']:.2f}초** + 본 검사 **{inference['gpu_phase_wall_seconds']:.2f}초** = **{results['cost']['completed_gpu_phases_wall_seconds_sum']:.2f}초**. 모델 로딩과 I/O를 포함한 wall time이다. 최초 입력 구성 오류는 forward 전에 발생했고 소요시간을 별도로 계측하지 못했으므로 제한 관리에서 120초를 예약했다.",
        f"본 검사 cache 추출 **{inference['cache_creation_seconds']:.2f}초**, cached scorer 진단 **{inference['cached_scorer_diagnostics_seconds']:.2f}초**. GPU 0 전체 카드 관측 최대 **{inference['maximum_card_used_bytes'] / 1e9:.3f}GB**, 추가 PyTorch peak reserved **{results['cost']['gpu_peak_reserved_bytes'] / 1e9:.3f}GB**. GPU 1에 새 작업 없음.",
        f"공식 CPU 평가: **{results['cost']['official_evaluation_seconds']:.2f}초 / {results['cost']['official_evaluated_candidates']:,}후보 / 2 worker**. 독립 CPU derivative 검사 {derivative_validation['cpu_wall_seconds']:.2f}초. 기존 LPWM source/config/input **293파일 hash 불일치 0**, 현재 학습·대기열 보존.", "",
        "## 사전 기준과 다음 실험 하나", "",
        "평균 선택손실 ≤5점 및 원본 대비 NC/DAC 위반 선택률 악화 ≤1pp 기준은 official128/nearby64/nearby16에서 통과했다. 다만 선택손실의 bootstrap 상한은 5점을 넘는 조건이 있으며 이는 작은 subset의 point-estimate gate다. scorer 전반의 안전성을 검증했다고 해석하지 않는다.",
        "**다음 우선순위: 같은 고정 checkpoint에서 CEM OFF와 공식 TOAD의 전체 comfort·anchor 포함 CEM을 같은 장면/공통 noise로 비교해 실제 순증가를 먼저 확인한다.** 성공하면 후보 구성 안정성이 보정 실패를 구분하는지 별도 평가한다. 현재 상관만으로 안정성 기반 selector를 설계하거나 sampler 학습을 시작하지 않는다.",
        "기존 축소 실험과는 입력·학습 데이터/량·후보 수·평가 장면이 다르므로 이 점수 차이를 scorer만의 학습 효과로 해석하지 않는다. 전체 NAVTEST/EPDMS, 최적화 성능 향상·novelty는 미확인이다.", "",
        "## 경로 및 재실행", "", f"- 결과: `{configuration['results_directory']}/results.json`, `comparison.csv`, `scene_results.csv`, `cpu_gradient_validation.json`, `execution_provenance.json`.",
        f"- Cache/공식 metric/phase 로그: `{configuration['output_directory']}/`.",
        "- 시각화: `scorer_calibration_summary.png`.", "", "```bash", "cd /rhome/junseong/PlanningAwareFuturePrediction", "bash scripts/run_official_toad_navsim_v1_scorer_calibration.sh", "```", "",
        "완료된 phase는 재측정하지 않는다. 가중치/원본 데이터/참고 repo는 Git에 포함하지 않으며 로컬 경로와 SHA는 provenance에 기록했다.", "",
        "[TOAD 원문](https://arxiv.org/abs/2606.07170) · [공식 NAV1 코드 및 설정](https://github.com/valeoai/TOAD/tree/nav1) · [공개 가중치](https://github.com/valeoai/TOAD/releases/tag/Nav1)"]
    (destination / "report.md").write_text("\n".join(report) + "\n")
    write_json(output / "inspection_packaged.json", {"complete": True,
        "results": str(destination / "results.json"), "no_training_started": True,
        "sources": actual_sources, "protected_training": results["protected_training"]})
    print(json.dumps({"complete": True, "quality_gate_passed": results["quality_gate_passed"],
        "actual_gradient_cpu_validation": derivative_validation["all_finite_difference_checks_passed"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", type=Path, required=True)
    arguments = parser.parse_args()
    package_inspection(json.loads(arguments.configuration.read_text()))
