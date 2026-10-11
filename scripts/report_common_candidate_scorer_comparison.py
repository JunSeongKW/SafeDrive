#!/usr/bin/env python
"""Ranking, calibration, safety errors and plots for identical candidates."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from calibrate_official_toad_navsim_v1_scorer import PROJECT_ROOT, protected_files_audit, write_json, file_sha256
from calibrate_frozen_drivor_scorer import METRIC_NAMES, recording_bootstrap


def report(configuration):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = PROJECT_ROOT / configuration["output_directory"]
    destination = PROJECT_ROOT / configuration["results_directory"]
    records = json.loads((output / "registration.json").read_text())["records"]
    inference = json.loads((output / "inference_complete.json").read_text())
    evaluation = json.loads((output / "official_evaluation_complete.json").read_text())
    arrays = np.load(output / "scorer_cache_and_candidates.npz")
    metrics = np.load(output / "official_metrics.npy")
    assert arrays["tokens"].tolist() == [record["token"] for record in records]
    assert metrics.shape == (64, 256, 7)
    grouping = [record["recording_group"] for record in records]
    scene_indices = np.arange(len(records))
    pools, scene_rows = {}, []
    plotting_labels = ["Oracle best", "DrivoR public", "TOAD-released DrivoR"]
    colors = ["#299348", "#2475c6", "#e98126"]
    for pool_name, (begin, end) in configuration["candidate_pools"].items():
        current_metrics = metrics[:, begin:end]
        current_pdms = current_metrics[..., -1] * 100
        oracle_indices = current_pdms.argmax(axis=1)
        oracle_selected = current_metrics[scene_indices, oracle_indices]
        oracle_pdms = oracle_selected[:, -1] * 100
        pool = {"candidate_count": end-begin, "oracle_best_pdms": float(oracle_pdms.mean()),
            "oracle_selected_submetrics": {name: float(oracle_selected[:, index].mean()) for index, name in enumerate(METRIC_NAMES[:-1])},
            "models": {}}
        selected_by_model = []
        for model_index, model_name in enumerate(configuration["models"]):
            log_scores = arrays["neural_log_scores"][model_index, :, begin:end]
            probabilities = arrays["metric_probabilities"][model_index, :, begin:end]
            estimate = np.exp(log_scores.astype(np.float64)) / 12 * 100
            selected_indices = log_scores.argmax(axis=1)
            selected = current_metrics[scene_indices, selected_indices]
            selected_pdms = selected[:, -1] * 100
            selected_by_model.append(selected_pdms)
            regret = oracle_pdms-selected_pdms
            correlations = [float(spearmanr(log_scores[index], current_pdms[index]).statistic) for index in scene_indices
                            if np.ptp(log_scores[index]) > 1e-12 and np.ptp(current_pdms[index]) > 1e-10]
            unsafe = (current_metrics[..., 0] < 1) | (current_metrics[..., 1] < 1)
            predicted_safe = (probabilities[..., 0] >= configuration["safe_prediction_threshold"]) & (probabilities[..., 1] >= configuration["safe_prediction_threshold"])
            falsely_safe = unsafe & predicted_safe
            row = {"selected_pdms": float(selected_pdms.mean()), "selection_regret_pdms_points": float(regret.mean()),
                "regret_recording_bootstrap_95ci": recording_bootstrap(regret, grouping, configuration["bootstrap_seed"], configuration["bootstrap_repetitions"]),
                "score_estimate_mae_pdms_points": float(np.abs(estimate-current_pdms).mean()),
                "score_estimate_rmse_pdms_points": float(np.sqrt(np.square(estimate-current_pdms).mean())),
                "within_scene_spearman_mean": float(np.mean(correlations)), "spearman_valid_scene_count": len(correlations),
                "unsafe_candidate_count": int(unsafe.sum()), "false_safe_candidate_count": int(falsely_safe.sum()),
                "false_safe_fraction_among_unsafe_candidates": float(falsely_safe.sum()/unsafe.sum()) if unsafe.any() else None,
                "unsafe_fraction_among_predicted_safe_candidates": float(falsely_safe.sum()/predicted_safe.sum()) if predicted_safe.any() else None,
                "selected_nc_failure_count": int((selected[:, 0] < 1).sum()), "selected_dac_failure_count": int((selected[:, 1] < 1).sum()),
                "selected_ttc_failure_count": int((selected[:, 3] < 1).sum()),
                "selected_submetrics": {name: float(selected[:, index].mean()) for index, name in enumerate(METRIC_NAMES[:-1])},
                "metric_mae": {name: float(np.abs(probabilities[..., index]-current_metrics[..., index]).mean()) for index, name in enumerate(METRIC_NAMES[:-1])},
                "metric_mean_squared_error": {name: float(np.square(probabilities[..., index]-current_metrics[..., index]).mean()) for index, name in enumerate(METRIC_NAMES[:-1])},
                "by_current_driving_command": {command: {"scenes": sum(record["command_name"] == command for record in records),
                    "selected_pdms": float(selected_pdms[[record["command_name"] == command for record in records]].mean()),
                    "oracle_best_pdms": float(oracle_pdms[[record["command_name"] == command for record in records]].mean())}
                    for command in ("left_turn", "straight", "right_turn")}}
            pool["models"][model_name] = row
            for scene_index, record in enumerate(records):
                scene_rows.append({"pool": pool_name, "model": model_name, "token": record["token"],
                    "recording_group": record["recording_group"], "command": record["command_name"],
                    "selected_index_in_pool": int(selected_indices[scene_index]),
                    "oracle_index_in_pool": int(oracle_indices[scene_index]),
                    "selected_pdms": float(selected_pdms[scene_index]), "oracle_pdms": float(oracle_pdms[scene_index]),
                    "selected_predicted_pdms": float(estimate[scene_index, selected_indices[scene_index]]),
                    "regret": float(regret[scene_index])})
        difference = selected_by_model[1]-selected_by_model[0]
        pool["toad_minus_drivor_selected_pdms"] = float(difference.mean())
        pool["toad_minus_drivor_recording_bootstrap_95ci"] = recording_bootstrap(difference, grouping, configuration["bootstrap_seed"], configuration["bootstrap_repetitions"])
        pools[pool_name] = pool
    summary = {"complete": True, "benchmark": "NAVSIM v1 NAVTEST64 development diagnostic",
        "scene_count": len(records), "recording_count": len(set(grouping)), "full_navtest": False,
        "independent_final_test": False, "identical_candidate_contexts_for_both_scorers": True,
        "no_cem_search_executed": True, "optimizer_steps": 0, "pools": pools,
        "score_estimate_definition": "100 * exp(neural_log_score) / 12 = soft NC*DAC*(5*EP+5*TTC+2*comfort)/12; an estimate, not the official oracle and not a certified probability",
        "checkpoints": {name: {"epoch_index": loading["checkpoint_epoch"], "completed_epochs": loading["checkpoint_epoch"]+1,
            "global_step": loading["checkpoint_global_step"], "strict_loading": loading["strict_loading"], "tensor_count": loading["checkpoint_tensor_count"]}
            for name, loading in inference["models"].items()},
        "checks": {"cached_encoder_calls": 0, "frozen_weights_unchanged": inference["all_weights_unchanged"],
            "official_pair_evaluator_maximum_error": evaluation["maximum_pair_evaluator_error"],
            "original_drivor_forward_maximum_error": max(row["original_drivor_forward_maximum_error"] or 0 for row in inference["checks"]["drivor_public"]),
            "scorer_and_decoder_sources_identical": inference["official_scorer_source_matches_original_drivor"] and inference["official_scorer_decoder_source_matches_original_drivor"]},
        "cost": {"gpu_phase_wall_seconds": inference["gpu_phase_wall_seconds"], "maximum_card_used_bytes": inference["maximum_card_used_bytes"],
            "additional_peak_reserved_bytes": inference["memory"]["peak_reserved_bytes"], "official_cpu_evaluation_seconds": evaluation["cpu_wall_seconds"],
            "official_evaluated_candidates": evaluation["official_evaluated_candidates"]},
        "protected_training": protected_files_audit(json.loads((PROJECT_ROOT / configuration["reference_configuration"]).read_text()))}
    write_json(destination / "results.json", summary)
    write_json(destination / "inference_validation.json", inference)
    write_json(destination / "oracle_validation.json", evaluation)
    with (destination / "scene_results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scene_rows[0])); writer.writeheader(); writer.writerows(scene_rows)
    with (destination / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=("pool", "model", "candidate_count", "selected_pdms", "oracle_best_pdms", "selection_regret_pdms_points", "score_estimate_mae_pdms_points", "within_scene_spearman_mean", "selected_nc_failure_count", "selected_dac_failure_count")); writer.writeheader()
        for name, pool in pools.items():
            for model_name, row in pool["models"].items():
                writer.writerow({"pool": name, "model": model_name, "candidate_count": pool["candidate_count"], "oracle_best_pdms": pool["oracle_best_pdms"], **{key: row[key] for key in writer.fieldnames if key in row}})
    figure, axes = plt.subplots(2, 3, figsize=(17, 10))
    for pool_index, (name, pool) in enumerate(pools.items()):
        values = [pool["oracle_best_pdms"]] + [pool["models"][name]["selected_pdms"] for name in configuration["models"]]
        axes[0, pool_index].bar(np.arange(3), values, color=colors)
        axes[0, pool_index].set_xticks(np.arange(3), ["Oracle", "DrivoR", "TOAD-release"])
        axes[0, pool_index].set_ylim(0, 106); axes[0, pool_index].set_ylabel("Actual PDMS of selected path")
        axes[0, pool_index].set_title(name + " | identical candidates")
        for index, value in enumerate(values):
            axes[0, pool_index].text(index, value+1.5, f"{value:.2f}", ha="center")
    begin, end = configuration["candidate_pools"]["balanced_native128"]
    for model_index, model_name in enumerate(configuration["models"]):
        estimate = np.exp(arrays["neural_log_scores"][model_index, :, begin:end].astype(np.float64)) / 12 * 100
        axes[1, model_index].scatter(estimate.ravel(), metrics[:, begin:end, -1].ravel()*100, s=4, alpha=.2, color=colors[model_index+1])
        axes[1, model_index].plot([0,100],[0,100], "k--", linewidth=1)
        axes[1, model_index].set_xlim(-2,102); axes[1, model_index].set_ylim(-2,102)
        axes[1, model_index].set_xlabel("Scorer PDMS estimate"); axes[1, model_index].set_ylabel("Official oracle PDMS")
        axes[1, model_index].set_title(model_name + " | diagonal = accurate estimate")
    names = METRIC_NAMES[:-1]
    positions = np.arange(len(names))
    for model_index, model_name in enumerate(configuration["models"]):
        axes[1, 2].bar(positions + (-.18 if model_index == 0 else .18),
            [pools["balanced_native128"]["models"][model_name]["metric_mae"][name] for name in names], width=.36,
            color=colors[model_index+1], label=plotting_labels[model_index+1])
    axes[1, 2].set_xticks(positions, ["NC", "DAC", "Progress", "TTC", "Comfort", "DDC"])
    axes[1, 2].set_ylabel("Metric MAE (0-1); lower is better"); axes[1, 2].legend(fontsize=8)
    axes[1, 2].set_title("Balanced native pool: submetric prediction errors")
    figure.suptitle("Frozen scorer comparison | NAVTEST64 development subset\nOracle selects the actual best candidate; TOAD-release is another DrivoR checkpoint, CEM OFF", fontsize=13)
    figure.tight_layout(rect=(0,0,1,.94)); figure.savefig(destination / "oracle_vs_public_scorers.png", dpi=160); plt.close(figure)
    lines = ["# 동일 후보에서 Oracle·DrivoR·TOAD 배포 scorer 비교", "",
        "## 결론과 범위", "",
        "공식 NAVSIM v1 oracle은 미래 GT와 지도 등을 이용하는 실제 평가기다. Oracle 행은 동일 후보 집합 중 실제 PDMS가 가장 높은 경로를 선택하는 오프라인 상한이며, PDMS를 항상 100으로 놓은 값이 아니다.",
        "TOAD에 별도의 새 scorer 구조는 없다. 비교 대상은 DrivoR 공개 Scaling 가중치와 TOAD에서 배포한 random-bouquet DrivoR 가중치다. 실제 scorer 및 decoder 코드가 같음을 확인했다. 모든 가중치 고정, 추가 학습/CEM 0.",
        "NAVTEST64 / 38 recording / 현재 명령 좌21·직22·우21의 이전 노출 개발 패널이다. 전체 NAVTEST·독립 최종 test가 아니다. 두 scorer 모두 동일한 전체 pool을 채점해 candidate self-attention context를 통제했다.", "",
        "## 실제 선택 결과", "", "| 동일 후보 pool | 후보 수 | Oracle 최고 PDMS | DrivoR 선택 | TOAD 배포본 선택 | DrivoR 선택손실 | TOAD 선택손실 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, pool in pools.items():
        first, second = [pool["models"][name] for name in configuration["models"]]
        lines.append(f"| {name} | {pool['candidate_count']} | {pool['oracle_best_pdms']:.4f} | {first['selected_pdms']:.4f} | {second['selected_pdms']:.4f} | {first['selection_regret_pdms_points']:.4f} | {second['selection_regret_pdms_points']:.4f} |")
    lines += ["", "Balanced native128는 각 모델이 생성한 64개 경로를 합친 공통 pool이다. Random64와 nearby64는 이전 공식 TOAD 진단의 동일 경로를 재사용했으므로 TOAD 기반 분포라는 한계가 있다. 후보 생성 품질과 scorer 선택 품질을 분리하기 위해 같은 pool의 최고 점수와 선택 점수 차이를 계산했다.", "",
        "## 예측값과 실제 값", "", "| 후보 pool | scorer | PDMS 환산 예측 MAE | 장면 내 Spearman | 선택 NC 위반 | 선택 DAC 위반 |", "|---|---|---:|---:|---:|---:|"]
    for name, pool in pools.items():
        for model_name, row in pool["models"].items():
            lines.append(f"| {name} | {model_name} | {row['score_estimate_mae_pdms_points']:.4f} | {row['within_scene_spearman_mean']:.4f} | {row['selected_nc_failure_count']}/64 | {row['selected_dac_failure_count']}/64 |")
    lines += ["", "환산 예측 = 100×NC확률×DAC확률×(5×progress+5×TTC+2×comfort)/12. Raw log score를 공식 PDMS와 같은 단위라고 취급하지 않았다. Soft submetric 조합은 실제 oracle의 시뮬레이션/threshold 처리와 다르며 안전 확률 보장이 아니다. NC·DAC·TTC·comfort 및 progress 예측오차, 0.9 안전 confidence에서 false-safe 비율, 명령별 점수와 recording-bootstrap CI는 results.json에 기록했다.", "",
        "## 실제 결과의 해석", "",
        "1. 공통 native128의 선택손실은 DrivoR 6.25점, TOAD 배포본 4.00점이다. 같은 pool 안에 더 좋은 경로가 있으므로 이 차이는 후보 부족보다 **scorer 선택의 오차**를 측정한다. 후보 내 oracle도 전체 가능한 경로의 최적값은 아니다.",
        "2. TOAD 배포본은 무작위64에서 +13.38점, 주변64에서 +5.44점 높았다. TOAD에서 유래한 두 후보 분포에서 원 DrivoR보다 더 잘 선택했다는 관측이다. Random-bouquet 학습 한 요소의 효과나 전체 CEM 성능을 입증하는 비교는 아니다.",
        "3. 두 모델 모두 안전 판단 오류가 남는다. Native128에서 두 모델은 각각 NC<1인 경로를 1/64회 선택했다. Random64에서는 NC<1 선택이 5→4회, DAC<1 선택이 6→0회였다. Nearby64에서는 DAC<1 선택이 3→0회였다. NC<1은 공식 지표의 부분 감점도 포함하며 모든 경우를 완전 충돌로 부르지 않는다.",
        "4. Native128 전체 후보의 환산 예측 MAE는 17.66→15.89점, 장면 내 순위 상관은 .430→.495였다. 순위 개선과 절대 점수 보정은 서로 다른 문제이며, 95.95의 선택 성능이 모든 경로 점수를 정확히 예측했다는 의미는 아니다.",
        "5. Native128에서 실제 NC<1 또는 DAC<1인 후보 1,294개 중, 두 안전 head가 모두 .9 이상인 후보는 DrivoR 298개(23.03%), TOAD 배포본 247개(19.09%)였다. 후보들이 서로 독립이라는 가정이나 이 비율을 실제 배포 사고 확률로 해석하지 않는다.", "",
        "### 같은 장면에 대한 대응 비교", "",
        "| 후보 pool | TOAD 배포본−DrivoR PDMS | Recording bootstrap 95% CI |", "|---|---:|---|" ]
    for pool_name, pool in pools.items():
        lower, upper = pool["toad_minus_drivor_recording_bootstrap_95ci"]
        lines.append(f"| {pool_name} | {pool['toad_minus_drivor_selected_pdms']:+.4f} | [{lower:+.4f}, {upper:+.4f}] |")
    lines += ["", "2000회 recording 단위 bootstrap이다. 이 패널의 표본 불확실성을 나타내며 학습 seed 변동이나 독립 최종 평가의 불확실성을 대신하지 않는다.", "",
        "### 현재 주행 명령별 native128 선택", "",
        "| 현재 명령 | 장면 수 | Oracle | DrivoR | TOAD 배포본 |", "|---|---:|---:|---:|---:|"]
    native_models = pools["balanced_native128"]["models"]
    for command in ("left_turn", "straight", "right_turn"):
        first, second = [native_models[name]["by_current_driving_command"][command] for name in configuration["models"]]
        lines.append(f"| {command} | {first['scenes']} | {first['oracle_best_pdms']:.4f} | {first['selected_pdms']:.4f} | {second['selected_pdms']:.4f} |")
    lines += ["", "현재 ego status에 저장된 명령을 이용했다. 결과를 보고 후보 성능으로 상황을 분류하지 않았다. 각 21–22장면으로 특정 상황의 일반적 우열을 단정하지 않는다.", "",
        "## 그림 해석", "",
        "- `oracle_vs_public_scorers.png`: 위쪽 초록은 같은 후보 중 실제 최고, 파랑은 DrivoR가 고른 경로, 주황은 TOAD 배포본이 고른 경로의 **실제 PDMS**다. 초록과 다른 막대 사이가 선택손실이다.",
        "- 아래 scatter는 x=모델 환산 예측, y=공식 실제 PDMS다. 대각선에서 멀수록 예측오차가 크다. 오른쪽 아래에 있는 점은 모델이 좋다고 예측했으나 실제 점수가 낮은 경로다.",
        "- `case_*_navsim.png`는 결과를 보고 선택한 오류 사례다. 초록은 인간 GT가 아니라 **후보 내 oracle 최고 경로**, 파랑/주황은 두 scorer 선택이다. 극단 사례를 일반적인 장면처럼 해석하지 않는다.", "",
        "## 통제·가중치·해석 한계", "",
        "- 같은 이미지4camera/현재1frame/1148×672/PILresize/ImageNet, 같은 현재ego11. 각 checkpoint 자신의 backbone feature64×256FP32를 사용한다. 서로 다른 모델의 feature를 임의 교환하지 않았다.",
        "- DrivoR asset은 파일명30epoch이나 metadata는 epoch28(29완료)/53911update; TOAD 배포본은 epoch29(30완료)/55770update다. 둘 다 배포 명시 trainval103k+Simscale134k지만 학습량·학습 recipe와 backbone 적응이 완전히 같은 통제는 아니다. 점수 차이를 random-bouquet 한 요소의 인과효과로 해석하지 않는다.",
        "- 기존 local 축소 DrivoR와 공개 모델을 구분한다. 이번 공식weights의 점수로 과거 local scorer의 수치를 대체하지 않는다. CEM ON/OFF 성능 비교는 수행하지 않았다.",
        f"- 467tensor strictload, 원 DrivoR forward와 compatibility forward 최대오차 {summary['checks']['original_drivor_forward_maximum_error']}, cache score 매칭/encoder 호출0, frozenweights 불변. 공식 oracle pair evaluator 검증 최대오차 {summary['checks']['official_pair_evaluator_maximum_error']}.", "",
        "## 실제 비용", "",
        f"GPU0 phase {summary['cost']['gpu_phase_wall_seconds']:.2f}초(load/I/O/cache포함), 전체 카드 관측 최대 {summary['cost']['maximum_card_used_bytes']/1e9:.3f}GB. 새 GPU1 작업 없음. 공식 CPU평가 {summary['cost']['official_cpu_evaluation_seconds']:.2f}초/{summary['cost']['official_evaluated_candidates']:,}후보/2worker. 보호293파일 hash 불일치0·LPWM 학습/queue 유지.", "",
        "## 재실행", "", "```bash", "cd /rhome/junseong/PlanningAwareFuturePrediction", "bash scripts/run_common_candidate_scorer_comparison.sh", "```", "",
        "[DrivoR 가중치](https://github.com/valeoai/DrivoR/releases/tag/Scaling) · [TOAD 배포 가중치](https://github.com/valeoai/TOAD/releases/tag/Nav1) · [TOAD method](https://arxiv.org/html/2606.07170v1#S3.SS2)"]
    visualization_manifest = destination / "visualization_manifest.json"
    if visualization_manifest.exists():
        visualization = json.loads(visualization_manifest.read_text())
        diagnostic_cases = []
        lines += ["", "## NAVSIM 오류 사례", ""]
        for case in visualization["cases"]:
            scene_index = arrays["tokens"].tolist().index(case["token"])
            candidate_offset = configuration["candidate_pools"][case["pool"]][0]
            diagnostic = {**case, "official_selected_submetrics": {}}
            for role, selected_index in zip(("drivor_public", "toad_released_drivor", "oracle_best"), case["selected_indices"]):
                values = metrics[scene_index, candidate_offset+selected_index]
                diagnostic["official_selected_submetrics"][role] = {name: float(values[index]) for index, name in enumerate(METRIC_NAMES)}
            diagnostic_cases.append(diagnostic)
            first, second, oracle = case["selected_actual_pdms"]
            lines.append(f"- [{case['pool']} / {case['token']}]({Path(case['image']).name}): 실제 PDMS(DrivoR/TOAD배포/Oracle) {first:.2f} / {second:.2f} / {oracle:.2f}.")
        write_json(destination / "case_diagnostics.json", {"examples_selected_after_scoring": True, "cases": diagnostic_cases})
    lines += ["", "## 다음 확인 하나", "",
        "같은 TOAD 공개 가중치·같은 장면·공통 random noise에서 CEM OFF와 전체 공식 CEM ON(comfort·anchor·fallback 포함)의 실제 PDMS와 oracle 선택손실을 비교한다. 현재 결과는 학습된 scorer의 경로 선택 오류가 남는다는 근거이며, 최적화 루프가 이를 줄이거나 악화시키는지는 아직 평가하지 않았다. 이 후속 작업은 자동 기동하지 않았다."]
    (destination / "report.md").write_text("\n".join(lines)+"\n")
    write_json(destination / "provenance.json", {"configuration": configuration, "inference_source_sha256": inference["source_sha256"],
        "report_source_sha256": file_sha256(Path(__file__)), "checkpoint_hashes": {"drivor": configuration["drivor_checkpoint_sha256"], "toad": json.loads((PROJECT_ROOT / configuration["reference_configuration"]).read_text())["checkpoint_sha256"]},
        "cache_path": str(output / "scorer_cache_and_candidates.npz"), "cache_sha256": file_sha256(output / "scorer_cache_and_candidates.npz"), "environment": inference["environment"]})
    write_json(output / "comparison_packaged.json", summary)
    print(json.dumps({"complete": True, "pools": pools}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", type=Path, required=True)
    arguments = parser.parse_args()
    report(json.loads(arguments.configuration.read_text()))
