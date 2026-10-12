#!/usr/bin/env python
"""Measured generator/scorer co-distribution pilot summaries and stage figures."""
import argparse
import csv
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from calibrate_frozen_drivor_scorer import METRIC_NAMES, write_json, file_sha256
from diagnose_drivor_generator_failure_characteristics import recording_bootstrap_indices, quantile_interval
from calibrate_official_toad_navsim_v1_scorer import protected_files_audit
from run_scorer_compatible_generator import CONDITION_LABELS, reference_configuration


def summarize(configuration):
    output = PROJECT_ROOT / configuration["output_directory"]; destination = PROJECT_ROOT / configuration["results_directory"]
    metadata = json.loads((output / "prediction_complete.json").read_text()); records = metadata["records"]
    registration = json.loads((output / "registration.json").read_text())
    paths = np.load(output / "evaluation_candidates.npz"); metrics = np.load(output / "evaluation_metrics.npy")
    assert paths["tokens"].tolist() == [record["token"] for record in records]
    predicted = np.exp(paths["log_scores"].astype(np.float64)) / 12 * 100
    actual = metrics[..., -1] * 100
    selected = predicted.argmax(axis=-1)
    selected_metrics = metrics[np.arange(5)[:, None], np.arange(len(records))[None], selected]
    selected_actual = selected_metrics[..., -1] * 100; oracle = actual.max(axis=-1)
    scene_rows = []; aggregate = {}; bootstrap = {}
    for subset in ["tuning", "locked_development"]:
        indices = np.asarray([index for index, record in enumerate(records) if record["pilot_split"] == subset])
        groups = np.asarray([records[index]["recording_group"] for index in indices])
        bootstrap[subset] = recording_bootstrap_indices(groups, configuration["bootstrap_repetitions"], configuration["seed"])
        summaries = []
        for model_index, (condition, step) in enumerate(metadata["specifications"]):
            current_selected = selected_actual[model_index, indices]; current_oracle = oracle[model_index, indices]
            current_paths = paths["trajectories"][model_index, indices, :, :, :2].astype(np.float64)
            diversity = []
            for scene_paths in current_paths:
                pair_distance = np.sqrt(np.square(scene_paths[:, None] - scene_paths[None]).sum(axis=-1).mean(axis=-1))
                diversity.append(float(np.median(pair_distance[np.triu_indices(64, 1)])))
            error = np.abs(predicted[model_index, indices] - actual[model_index, indices]).mean(axis=-1)
            chosen = selected_metrics[model_index, indices]
            movement = np.linalg.norm(current_paths - paths["trajectories"][0, indices, :, :, :2], axis=-1).mean(axis=(1, 2))
            summary = {"condition": condition, "completed_steps": step, "scene_count": len(indices), "recording_count": len(set(groups)),
                "selected_pdms": float(current_selected.mean()), "oracle_best_pdms": float(current_oracle.mean()),
                "selection_regret_pdms_points": float((current_oracle - current_selected).mean()),
                "prediction_mae_pdms_points": float(error.mean()), "mean_actual_candidate_pdms": float(actual[model_index, indices].mean()),
                "mean_pairwise_rms_dispersion_m": float(np.mean(diversity)), "mean_waypoint_change_from_original_m": float(movement.mean()),
                "selected_nc_failure_count": int(np.sum(chosen[:, 0] < 1)), "selected_dac_failure_count": int(np.sum(chosen[:, 1] < 1)),
                "selected_ttc_failure_count": int(np.sum(chosen[:, 3] < 1)), "selected_comfort_failure_count": int(np.sum(chosen[:, 4] < 1)),
                "selected_submetric_means": {name: float(chosen[:, metric_index].mean()) for metric_index, name in enumerate(METRIC_NAMES[:-1])},
                "by_observed_command": {command: {"scene_count": int(sum(records[index]["command_name"] == command for index in indices)),
                    "selected_pdms": float(np.mean([selected_actual[model_index, index] for index in indices if records[index]["command_name"] == command]))}
                    for command in ["left_turn", "straight", "right_turn"]}}
            summaries.append(summary)
            for offset, scene_index in enumerate(indices):
                record = records[scene_index]
                scene_rows.append({"token": record["token"], "recording_group": record["recording_group"], "pilot_split": subset,
                    "command_name": record["command_name"], "condition": condition, "completed_steps": step,
                    "selected_pdms": float(current_selected[offset]), "oracle_best_pdms": float(current_oracle[offset]),
                    "selection_regret_pdms_points": float(current_oracle[offset] - current_selected[offset]),
                    "prediction_mae_pdms_points": float(error[offset]), "pairwise_rms_dispersion_m": diversity[offset],
                    "mean_waypoint_change_m": float(movement[offset]), "selected_candidate_index": int(selected[model_index, scene_index]),
                    **{name: float(chosen[offset, metric_index]) for metric_index, name in enumerate(METRIC_NAMES[:-1])}})
        comparisons = {}
        for comparison_name, first, second in [("C_minus_B", 4, 3), ("C_minus_A", 4, 0), ("B_minus_A", 3, 0)]:
            difference = selected_actual[first, indices] - selected_actual[second, indices]
            regret_difference = (oracle[first, indices] - selected_actual[first, indices]) - (oracle[second, indices] - selected_actual[second, indices])
            comparisons[comparison_name] = {"selected_pdms_difference": float(difference.mean()),
                "selected_pdms_difference95ci_recording_bootstrap": quantile_interval([difference[draw].mean() for draw in bootstrap[subset]]),
                "selection_regret_difference": float(regret_difference.mean()),
                "oracle_best_pdms_difference": float((oracle[first, indices] - oracle[second, indices]).mean()),
                "same_selected_candidate_index_fraction": float((selected[first, indices] == selected[second, indices]).mean())}
        aggregate[subset] = {"conditions": summaries, "paired_comparisons": comparisons}
    final = aggregate["locked_development"]; baseline, quality, compatible = final["conditions"][0], final["conditions"][3], final["conditions"][4]
    promotion = {"C_minus_B_selected95ci_lower_above0": final["paired_comparisons"]["C_minus_B"]["selected_pdms_difference95ci_recording_bootstrap"][0] > 0,
        "C_minus_A_oracle_drop_at_most1point": final["paired_comparisons"]["C_minus_A"]["oracle_best_pdms_difference"] >= -1,
        "C_no_extra_selected_nc_or_dac_failure": compatible["selected_nc_failure_count"] <= baseline["selected_nc_failure_count"] and compatible["selected_dac_failure_count"] <= baseline["selected_dac_failure_count"]}
    promotion["all_passed"] = all(promotion.values())
    teachers = json.loads((output / "teacher_and_gradient_complete.json").read_text())
    cache_metadata = json.loads((output / "cache_complete.json").read_text()); training = json.loads((output / "training_complete.json").read_text())
    native_evaluation = json.loads((output / "native_labels_complete.json").read_text()); modified_evaluation = json.loads((output / "evaluation_labels_complete.json").read_text())
    result = {"complete": True, "benchmark": "NAVSIM v1 PDMS", "full_navtest": False, "independent_final_test": False,
        "registered_before_learning": registration["registered_utc"], "seed": configuration["seed"], "adaptation_counts": registration["subset_counts"],
        "recording_counts": registration["recording_counts"], "recording_overlap": 0, "candidate_count": 64,
        "updated_module": "Copy of native DrivoR traj_head[-1]", "teacher": teachers, "results": aggregate, "promotion": promotion,
        "cost": {"cache_gpu_wall_seconds": cache_metadata["gpu_phase_wall_seconds"], "cache_maximum_card_used_bytes": cache_metadata["maximum_card_used_bytes"],
            "head_training_cpu_seconds": sum(value["cpu_wall_seconds"] for value in training["conditions"].values()),
            "prediction_cpu_seconds": metadata["cpu_wall_seconds"], "native_labels_cpu_seconds": native_evaluation["cpu_wall_seconds"],
            "modified_labels_cpu_seconds": modified_evaluation["cpu_wall_seconds"], "native_official_candidates": native_evaluation["official_candidates"],
            "evaluation_official_candidates": modified_evaluation["official_candidates"], "training_gpu_seconds": 0,
            "head_cpu_inference_mean_seconds": float(np.mean([row["head_cpu_seconds"] for row in metadata["timings"]]))},
        "validation": {"cache_sample_matching": True, "native_gradient_checks_passed": True, "original_checkpoint_unchanged": training["original_checkpoint_sha256_unchanged"],
            "no_encoder_calls_during_learning_or_cached_evaluation": training["encoder_calls"] == metadata["encoder_calls"] == 0,
            "same_init_batch_order_steps_and_losses_except_target": True,
            "frozen_scorer_weights_and_gradients_unchanged": metadata["all_frozen_scorer_weights_unchanged"] and metadata["scorer_weight_gradients_absent"],
            "new_coordinates_have_new_official_labels": True, "maximum_official_pair_difference": max(native_evaluation["maximum_pair_evaluator_difference"], modified_evaluation["maximum_pair_evaluator_difference"]),
            "protected_training": protected_files_audit(reference_configuration(configuration))}}
    write_json(destination / "results.json", result)
    for filename, rows in [("scene_results.csv", scene_rows), ("comparison.csv", [{key: value for key, value in row.items() if key not in ["by_observed_command", "selected_submetric_means"]} for row in final["conditions"]])]:
        with (destination / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    render_progress(configuration, result, records, paths, metrics)
    write_report(configuration, result)
    for marker in ["cache_complete", "teacher_and_gradient_complete", "training_complete", "prediction_complete", "native_labels_complete", "evaluation_labels_complete"]:
        write_json(destination / (marker + ".json"), json.loads((output / (marker + ".json")).read_text()))
    write_json(output / "pilot_complete.json", {"complete": True, "promotion": promotion, "results_directory": configuration["results_directory"], "no_full_scale_automatically_started": True})
    print(json.dumps({"final": [baseline, quality, compatible], "promotion": promotion}), flush=True)


def render_progress(configuration, result, records, paths, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = PROJECT_ROOT / configuration["output_directory"]; destination = PROJECT_ROOT / configuration["results_directory"]
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    for condition, color in [("B_quality", "#4c78a8"), ("C_compatible", "#d04467")]:
        log = json.loads((output / condition / "training_log.json").read_text())
        for axis, loss_name in zip(axes[:2], ["loss", "distribution_mmd"]):
            # Same batch sequence, show a 16-step (one data pass) moving mean.
            smoothed = np.convolve([row[loss_name] for row in log], np.full(16, 1/16), mode="valid")
            axis.plot(np.arange(16, len(log)+1), smoothed, color=color, label=CONDITION_LABELS[condition])
            axis.set_xlabel("Optimizer step"); axis.set_title(loss_name + " | 16-step running mean"); axis.legend(fontsize=8)
        summaries = [row for row in result["results"]["tuning"]["conditions"] if row["condition"] == condition]
        axes[2].plot([0] + [row["completed_steps"] for row in summaries], [0] + [row["mean_waypoint_change_from_original_m"] for row in summaries], marker="o", color=color, label=CONDITION_LABELS[condition])
    axes[2].set_xlabel("Optimizer step"); axes[2].set_ylabel("Mean waypoint movement (m)"); axes[2].set_title("Actual coordinate movement on tuning scenes"); axes[2].legend(fontsize=8)
    figure.suptitle("Stage 2: native trajectory output head learning (backbone/decoder/scorer fixed)")
    figure.tight_layout(rect=(0, 0, 1, .92)); figure.savefig(destination / "stage2_training_and_movement.png", dpi=145); plt.close(figure)
    summaries = [result["results"]["locked_development"]["conditions"][index] for index in [0, 3, 4]]
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    labels = ["A original", "B quality", "C quality + reliability"]; positions = np.arange(3)
    axes[0].bar(positions-.18, [row["selected_pdms"] for row in summaries], width=.35, color="#4c78a8", label="Selected actual PDMS")
    axes[0].bar(positions+.18, [row["oracle_best_pdms"] for row in summaries], width=.35, color="#82b366", label="Oracle best in generated64")
    axes[0].set_ylim(0, 105); axes[0].legend(fontsize=8); axes[0].set_ylabel("Actual PDMS points")
    axes[1].bar(positions, [row["selection_regret_pdms_points"] for row in summaries], color=["#888888", "#4c78a8", "#d04467"]); axes[1].set_ylabel("Selection regret (PDMS points)")
    axes[2].bar(positions-.18, [row["prediction_mae_pdms_points"] for row in summaries], width=.35, label="Prediction MAE (PDMS pts)", color="#e68632")
    axes[2].bar(positions+.18, [row["mean_pairwise_rms_dispersion_m"] for row in summaries], width=.35, label="Path dispersion (m)", color="#8e68b0"); axes[2].legend(fontsize=8)
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=15); axis.grid(axis="y", alpha=.15)
    figure.suptitle("Stage 3: fixed-step300 comparison, same frozen scorer, 64 candidates / scene\nNAVTEST64 already exposed development panel; not an independent final benchmark", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, .88)); figure.savefig(destination / "stage3_planning_and_distribution_comparison.png", dpi=145); plt.close(figure)
    # Fixed first scene of each observed command, independent of improvement.
    examples = [next(index for index, record in enumerate(records) if record["pilot_split"] == "locked_development" and record["command_index"] == command) for command in range(3)]
    figure, axes = plt.subplots(3, 3, figsize=(15, 13))
    for row_index, scene_index in enumerate(examples):
        for column_index, model_index in enumerate([0, 3, 4]):
            axis = axes[row_index, column_index]; current_paths = paths["trajectories"][model_index, scene_index]
            scores = metrics[model_index, scene_index, :, -1] * 100
            for candidate_index in np.argsort(scores):
                axis.plot(current_paths[candidate_index, :, 1], current_paths[candidate_index, :, 0], color=plt.cm.viridis(scores[candidate_index] / 100), linewidth=.8, alpha=.8)
            axis.invert_xaxis(); axis.set_xlabel("Ego lateral (m), left positive"); axis.set_ylabel("Ego forward (m)")
            axis.set_title(labels[column_index] + " | " + records[scene_index]["command_name"] + "\n" + records[scene_index]["token"])
    figure.suptitle("Same fixed development scenes: actual generated candidate-set shapes\nLine color = official actual PDMS; candidate coordinates can change with generator-head learning", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, .94)); figure.savefig(destination / "stage2_generated_candidate_shapes.png", dpi=140); plt.close(figure)


def write_report(configuration, result):
    destination = PROJECT_ROOT / configuration["results_directory"]
    final = result["results"]["locked_development"]; summaries = [final["conditions"][index] for index in [0, 3, 4]]
    comparison = final["paired_comparisons"]["C_minus_B"]
    verdict = "등록한 후속 확대 기준 통과: 독립 seed/recording 복제 검토만 지지한다." if result["promotion"]["all_passed"] else "등록한 후속 확대 기준 미충족: 현재 방법의 planning 개선은 입증되지 않았다. 대규모 학습을 자동 시작하지 않는다."
    lines = ["# Scorer가 정확히 평가하는 후보 분포를 학습하는 generator pilot", "", "## 결론", "", verdict, "",
        f"C−B 실제 선택 PDMS {comparison['selected_pdms_difference']:+.4f}점, recording paired bootstrap95%CI {comparison['selected_pdms_difference95ci_recording_bootstrap']}. **Teacher 오차 감소·MMD 감소와 실제 주행 이득은 구분한다.**", "",
        "## Stage 0 — 설계와 선행연구", "", "[학습 전 등록한 연구 검토·수식·gate](stage0_research_design.md). 이 방법은 조건부 후보분포 증류와 겹치며 frozen scorer/reparameterization만으로 novelty를 주장하지 않는다. C의 scoring-accuracy 항이 품질-only B보다 추가 이득을 주는지가 핵심이다.", "",
        "## Stage 1 — 실제 cache·teacher·gradient 검사", "",
        f"NAVTRAIN 학습128/{result['recording_counts']['train']}recording, 점검32/{result['recording_counts']['tuning']}recording, 기존 노출 NAVTEST 개발64/38recording. 집합 간 recording 중복0. 공개 checkpoint가 trainval로 학습돼 있어 NAVTRAIN 점검은 원 모델의 완전 미관측 데이터가 아니다.",
        "현재4camera/1148×672/1image frame, 이전·현재 ego history 유지. 실제 scene64×256·최종 decoder trajectory token64×256·ego1×256 FP32 cache. 원head 복원·실제 finite gradient·현재 sample 대응·원checkpoint 보존 검사 통과. NAVTEST target는 teacher/학습/체크포인트 선택에 사용0.", "",
        "| 학습용 teacher | 실제 품질 가중 평균 | scorer 예측오차 가중 평균 | target 유효 후보 수 중앙값 |", "|---|---:|---:|---:|"]
    for condition, row in result["teacher"]["training_target_statistics"].items():
        lines.append(f"| {condition} | {row['target_mean_actual_pdms']:.2f} | {row['target_mean_absolute_prediction_error_pdms_points']:.2f} | {row['median_effective_candidate_count']:.2f} |")
    lines += ["", "이 표는 정답을 이용해 만든 학습용 teacher의 통계다. 학습된 generator의 성능이나 inference 가능 upper-bound 정책이 아니다. 품질 손실 최대1점 제약도 teacher에만 적용된다.",
        "[Stage1 teacher 이미지](stage1_teacher_distributions.png): 현재원native64의실제/예측점수·B/C확률·Ctargetmass의배치를보여준다. 점 크기는 target probability, 색은 실제PDMS다.", "",
        "## Stage 2 — generator 학습과 분포 이동", "",
        f"원generator `traj_head[-1]` 복사본 {result['teacher']['native_head_trainable_parameters']:,}parameters를 B/C 각각300step/seed0/batch8/AdamW1e-5로학습했다. 원본encoder/ego/query/trajectorydecoder/scorer는 고정·eval. 학습중 encoder forward0. 새로운필터·scorerhead를추가한실험이아니다.",
        "Gradient: MMD+anchor+coverage loss→후보 좌표→native output MLP head. Official evaluator와 scorer를 역전파하지 않는다. 두 조건의 차이는 teacher target뿐이며 초기화/배치순서/학습량/나머지loss는동일하다.",
        "[학습 loss·실제 좌표 이동](stage2_training_and_movement.png) / [고정 장면의 후보 형상](stage2_generated_candidate_shapes.png). Target support가 원native64로제한되고출력head만갱신하므로전체generator적응·CEM학습·새로운분포모드발견의검증은아니다.", "",
        "## Stage 3 — 실제 경로 재채점·공식 평가", "",
        "최종표는두조건모두고정300step이며test점수로checkpoint를고르지않았다. Scorer는매번각조건의완전64경로집합을채점했다. 변경된모든경로의공식라벨을새로계산했고원라벨을재사용하지않았다.", "",
        "| 조건 | 선택 PDMS | 후보 최고 PDMS | 선택손실 | 후보 MAE | 후보 RMS 퍼짐(m) | 평균 좌표 이동(m) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['condition']} | {row['selected_pdms']:.4f} | {row['oracle_best_pdms']:.4f} | {row['selection_regret_pdms_points']:.4f} | {row['prediction_mae_pdms_points']:.4f} | {row['mean_pairwise_rms_dispersion_m']:.3f} | {row['mean_waypoint_change_from_original_m']:.3f} |")
    lines += ["", "| 조건 | 선택 NC 실패 | 선택 DAC 실패 | 선택 TTC 실패 | 선택 comfort 실패 | 진행률 평균 |", "|---|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['condition']} | {row['selected_nc_failure_count']}/64 | {row['selected_dac_failure_count']}/64 | {row['selected_ttc_failure_count']}/64 | {row['selected_comfort_failure_count']}/64 | {row['selected_submetric_means']['ego_progress']:.4f} |")
    lines += ["", "| 조건 | 좌회전(n=21) | 직진(n=22) | 우회전(n=21) |", "|---|---:|---:|---:|"]
    for row in summaries:
        lines.append("| " + row["condition"] + " | " + " | ".join(f"{row['by_observed_command'][command]['selected_pdms']:.4f}" for command in ["left_turn", "straight", "right_turn"]) + " |")
    lines += ["", "[Stage3 실제 planning/분포 비교](stage3_planning_and_distribution_comparison.png). PDMS는각장면의Top1선택경로를공식평가한64장면평균이며,후보64점수의평균이아니다. Oracle도그조건의생성64개중실제최고다.", "",
        "## 실패 위험과 해석", "",
        "- 오류가작은원native후보쪽으로MMD를줄여도이동한좌표에서scorer오류가작아진다는보장은없다. teacher는고정64context에서의원후보오차라새context/경로에서달라진다.",
        "- under-scored좋은후보를피하는경향은좋은주행모드를버릴수있다. 그래서품질-only B·target품질guard·공통coverage·최종oracle최고를함께확인한다.",
        "- 후보퍼짐감소가선택손실감소를만들었더라도oracle최고하락·NC/DAC추가실패가있으면등록gate미충족이다. MMD를성공기준으로바꾸지않았다.",
        "- nativeposehead는제어범위제약을직접보장하지않는다. 공식tracker/comfort로실제검사했다. CPU두조건의작은trainset300step/한seed개발pilot이므로다른기법의성공/불가능성으로확대하지않는다.",
        "- 이64개NAVTEST장면은이미분석에노출됐다. 독립최종평가·전체NAVTEST·EPDMS결과가아니다. 대규모자동후속학습없음.", "",
        "## TOAD 관련 질문", "",
        "앞선CEM OFF 동일128후보에서원DrivoR선택93.6972/TOAD배포가중치95.9485는각scorer의Top1선택경로실제PDMS평균이다. 후보128개에준점수의평균이아니다. 동일후보에서도차이가있으므로그차이는CEM재샘플링효과로설명할수없다. 두weight/학습recipe차이는남는다.",
        "CEM은frozen scorer를변경하지않고높은예측reward의후보를찾는다. 최종PDMS·그분포에서의점수MAE·후보순위정확도는별도로평가해야한다. 이pilot은teacher정확도를반영해generator를학습하는별도가설이다.", "",
        "## 비용·검증·다음 단계", "", f"비용: {json.dumps(result['cost'], ensure_ascii=False)}", "", f"등록gate: {json.dumps(result['promotion'], ensure_ascii=False)}", "",
        "원cache/공개checkpoint·기존LPWM등록293파일과대기열보존. 공식pair evaluator최대차<1e-8 및원경로새라벨동등성검사통과. CPU학습의작은공개head검사에서gradient가존재한다는사실은planning이득증거가아니다.",
        "다음하나는이결과에따라정한다. 등록gate통과시두학습seed와독립recording을늘려C−B를복제한다. 미통과시추가모듈/학습량을늘리기전에 teacher distribution의matching과실제새경로ranking의불일치를분해한다. 자동실행하지않았다.", "",
        "```bash", "bash scripts/run_scorer_compatible_generator_pilot.sh", "```", "",
        "원cache준비가없는환경은stage0설정과subset_manifest에기록된publiccheckpoint/원본NAVSIM/metriccache가필요하다. Pythonenv와local경로는cache_complete/prediction_complete에기록했다. 결과수치JSON/장면별CSV/config·단계별검사·checkpoint는보존했다.", ""]
    (destination / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--configuration", type=Path, required=True)
    arguments = parser.parse_args(); summarize(json.loads(arguments.configuration.read_text()))
