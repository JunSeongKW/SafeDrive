#!/usr/bin/env python
"""Additional accuracy-only generator condition; preserve completed A/B/C."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from calibrate_frozen_drivor_scorer import file_sha256, write_json, METRIC_NAMES
from run_scorer_compatible_generator import inputs, native_head, reference_configuration, evaluate_official_jobs
from calibrate_official_toad_navsim_v1_scorer import protected_files_audit
from diagnose_drivor_generator_failure_characteristics import recording_bootstrap_indices, quantile_interval


def load_source(configuration):
    source_configuration = json.loads((PROJECT_ROOT / configuration["source_configuration"]).read_text())
    source_output, registration, cache = inputs(source_configuration)
    return source_configuration, source_output, registration, cache


def train(configuration):
    import torch
    from planning_aware_future_prediction.models.scorer_compatible_generator import distribution_matching_losses
    source_configuration, source_output, source_registration, cache = load_source(configuration)
    output = PROJECT_ROOT / configuration["output_directory"]; destination = PROJECT_ROOT / configuration["results_directory"]
    labels = np.load(source_output / "native_metrics_all.npy")
    row_ids = [record["pilot_row"] for record in source_registration["records"] if record["pilot_split"] == "train"]
    source_batch_order = json.loads((source_output / "batch_order.json").read_text())
    assert row_ids == source_batch_order["training_rows"] and len(source_batch_order["batches"]) == configuration["training_steps"]
    torch.set_num_threads(source_configuration["cpu_threads"]); torch.use_deterministic_algorithms(True); torch.manual_seed(configuration["seed"])
    native_paths = torch.from_numpy(np.array(cache["native_trajectories"][row_ids])); tokens = torch.from_numpy(np.array(cache["trajectory_tokens"][row_ids]))
    actual = torch.from_numpy(labels[row_ids, :, -1].astype(np.float32)); predicted = torch.from_numpy(np.exp(np.array(cache["native_log_scores"][row_ids])) / 12)
    absolute_error = (predicted - actual).abs(); target = (-absolute_error / configuration["reliability_temperature"]).softmax(dim=1)
    assert torch.isfinite(target).all() and torch.allclose(target.sum(dim=1), torch.ones(len(target)))
    registration = {"registered_utc": datetime.now(timezone.utc).isoformat(), "configuration": configuration,
        "source_configuration": source_configuration, "records": source_registration["records"], "train_row_ids": row_ids,
        "source_cache_sha256": json.loads((source_output / "cache_complete.json").read_text())["cache_sha256"],
        "source_checkpoint_sha256": source_registration["checkpoint_sha256"], "source_native_head_sha256": file_sha256(source_output / "native_head_initialization.pt"),
        "source_batch_order_sha256": file_sha256(source_output / "batch_order.json"), "training_source_sha256": file_sha256(Path(__file__)),
        "navtest_rows_in_teacher": 0, "protected_training": protected_files_audit(reference_configuration(source_configuration))}
    write_json(output / "registration.json", registration); write_json(destination / "subset_manifest.json", registration); write_json(destination / "configuration.json", configuration)
    np.save(output / "accuracy_only_teacher_weights.npy", target.numpy())
    statistics = {"target_mean_actual_pdms": float((target * actual).sum(dim=1).mean() * 100),
        "target_mean_absolute_prediction_error_pdms_points": float((target * absolute_error).sum(dim=1).mean() * 100),
        "median_effective_candidate_count": float(target.square().sum(dim=1).reciprocal().median()),
        "target_mass_on_actual_pdms_below50": float((target * (actual < .5)).sum(dim=1).mean()),
        "target_mass_on_actual_pdms_zero": float((target * (actual == 0)).sum(dim=1).mean())}
    head = native_head(source_configuration).train(); optimizer = torch.optim.AdamW(head.parameters(), lr=configuration["learning_rate"], weight_decay=0.)
    started = time.perf_counter(); logs = []
    for step, rows in enumerate(source_batch_order["batches"], start=1):
        generated = head(tokens[rows]).reshape(len(rows), 64, 8, 3)
        # The unused coverage value is never added to the D objective. No quality
        # weights/guard are supplied; target depends solely on measured error.
        losses = distribution_matching_losses(generated, native_paths[rows], target[rows], target[rows], source_configuration["kernel_bandwidths_m"], source_configuration["heading_distance_scale_m"], source_configuration["coverage_temperature_m2"])
        loss = losses["distribution_mmd"] + configuration["anchor_loss_weight"] * losses["native_anchor"]
        optimizer.zero_grad(set_to_none=True); loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), source_configuration["gradient_clip_norm"])
        assert torch.isfinite(loss) and torch.isfinite(gradient_norm) and gradient_norm > 0
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in head.parameters())
        optimizer.step()
        logs.append({"step": step, "loss": float(loss.detach()), "distribution_mmd": float(losses["distribution_mmd"].detach()),
            "native_anchor": float(losses["native_anchor"].detach()), "gradient_norm": float(gradient_norm), "wall_seconds": time.perf_counter() - started})
        if step in configuration["checkpoint_steps"]:
            torch.save({"state_dict": head.state_dict(), "optimizer": optimizer.state_dict(), "completed_steps": step, "condition": "D_accuracy_only"}, output / f"step{step}.pt")
        if step % 100 == 0:
            print(json.dumps(logs[-1]), flush=True)
    write_json(output / "training_log.json", logs)
    write_json(output / "training_complete.json", {"complete": True, "training_target_statistics": statistics,
        "cpu_wall_seconds": time.perf_counter() - started, "parameters": sum(parameter.numel() for parameter in head.parameters()),
        "steps": configuration["training_steps"], "all_step_gradients_finite": True, "same_init_and_source_batch_order": True,
        "encoder_calls": 0, "gpu_seconds": 0, "quality_weight_or_guard_or_coverage_used": False})
    print(json.dumps({"teacher": statistics, "steps": len(logs)}), flush=True)


def predict(configuration):
    import torch
    from compare_public_drivor_toad_scorers import load_official_model, model_configuration_for, score_metrics
    from diagnose_residual_control_sampler import module_digest
    source_configuration, source_output, registration, cache = load_source(configuration)
    output = PROJECT_ROOT / configuration["output_directory"]
    comparison = json.loads((PROJECT_ROOT / source_configuration["comparison_configuration"]).read_text())
    model, loading = load_official_model(model_configuration_for(comparison, "drivor_public"), "cpu")
    torch.set_num_threads(source_configuration["cpu_threads"]); digest = module_digest(model)
    def reject_encoder(_module, _arguments):
        raise AssertionError("D cached inference must never execute image backbone")
    guard = model.image_backbone.register_forward_pre_hook(reject_encoder)
    records = json.loads((source_output / "prediction_complete.json").read_text())["records"]
    paths, scores, probabilities = [], [], []; started = time.perf_counter()
    with torch.inference_mode():
        for step in configuration["checkpoint_steps"]:
            head = native_head(source_configuration).eval(); head.load_state_dict(torch.load(output / f"step{step}.pt", map_location="cpu", weights_only=True)["state_dict"], strict=True)
            current_paths, current_scores, current_probabilities = [], [], []
            for record in records:
                row = record["pilot_row"]
                generated = head(torch.from_numpy(np.array(cache["trajectory_tokens"][row:row+1]))).reshape(1, 64, 8, 3)
                ranking, probability, _ = score_metrics(model, generated, torch.from_numpy(np.array(cache["scene_features"][row:row+1])), torch.from_numpy(np.array(cache["ego_tokens"][row:row+1])))
                current_paths.append(generated[0].numpy()); current_scores.append(ranking[0].numpy()); current_probabilities.append(probability[0].numpy())
            paths.append(current_paths); scores.append(current_scores); probabilities.append(current_probabilities)
    guard.remove(); assert digest == module_digest(model) and all(parameter.grad is None for parameter in model.parameters())
    np.savez_compressed(output / "evaluation_candidates.npz", tokens=[record["token"] for record in records], trajectories=np.asarray(paths), log_scores=np.asarray(scores), metric_probabilities=np.asarray(probabilities))
    write_json(output / "prediction_complete.json", {"complete": True, "records": records, "checkpoint_steps": configuration["checkpoint_steps"],
        "cpu_wall_seconds": time.perf_counter() - started, "scorer_frozen_unchanged_and_no_gradients": True, "encoder_calls": 0, "gpu_seconds": 0})


def labels(configuration):
    source_configuration, _, _, _ = load_source(configuration)
    output = PROJECT_ROOT / configuration["output_directory"]; metadata = json.loads((output / "prediction_complete.json").read_text())
    arrays = np.load(output / "evaluation_candidates.npz"); records = metadata["records"]
    jobs = [(record, arrays["trajectories"][:, index].reshape(-1, 8, 3), arrays["log_scores"][:, index].reshape(-1)) for index, record in enumerate(records)]
    measured = evaluate_official_jobs(source_configuration, jobs, output / "evaluation_metrics_flat.npy", output / "evaluation_progress.json")
    values = np.load(output / "evaluation_metrics_flat.npy").reshape(len(records), 2, 64, 7).transpose(1, 0, 2, 3)
    np.save(output / "evaluation_metrics.npy", values); write_json(output / "evaluation_complete.json", {"complete": True, **measured})


def report(configuration):
    source_configuration, source_output, registration, _ = load_source(configuration)
    output = PROJECT_ROOT / configuration["output_directory"]; destination = PROJECT_ROOT / configuration["results_directory"]
    source_result = json.loads((PROJECT_ROOT / source_configuration["results_directory"] / "results.json").read_text())
    source_predictions = np.load(source_output / "evaluation_candidates.npz"); source_metrics = np.load(source_output / "evaluation_metrics.npy")
    arrays = np.load(output / "evaluation_candidates.npz"); metrics = np.load(output / "evaluation_metrics.npy")
    metadata = json.loads((output / "prediction_complete.json").read_text()); records = metadata["records"]
    assert source_predictions["tokens"].tolist() == arrays["tokens"].tolist() == [record["token"] for record in records]
    training = json.loads((output / "training_complete.json").read_text())
    condition_reports = []; rows_for_csv = []
    for subset in ["tuning", "locked_development"]:
        indices = np.asarray([index for index, record in enumerate(records) if record["pilot_split"] == subset])
        groups = np.asarray([records[index]["recording_group"] for index in indices]); bootstrap = recording_bootstrap_indices(groups, source_configuration["bootstrap_repetitions"], configuration["seed"])
        for model_index, step in enumerate(configuration["checkpoint_steps"]):
            actual = metrics[model_index, indices, :, -1] * 100; predicted = np.exp(arrays["log_scores"][model_index, indices].astype(np.float64)) / 12 * 100
            chosen = predicted.argmax(-1); chosen_metrics = metrics[model_index, indices, chosen]
            selected_pdms = actual[np.arange(len(indices)), chosen]; oracle = actual.max(axis=1)
            current_paths = arrays["trajectories"][model_index, indices, :, :, :2].astype(np.float64)
            movement = np.linalg.norm(current_paths - source_predictions["trajectories"][0, indices, :, :, :2], axis=-1).mean(axis=(1, 2))
            dispersions = [float(np.median(np.sqrt(np.square(paths[:, None] - paths[None]).sum(-1).mean(-1))[np.triu_indices(64, 1)])) for paths in current_paths]
            row = {"condition": "D_accuracy_only", "subset": subset, "completed_steps": step, "scene_count": len(indices),
                "selected_pdms": float(selected_pdms.mean()), "oracle_best_pdms": float(oracle.mean()), "selection_regret_pdms_points": float((oracle - selected_pdms).mean()),
                "prediction_mae_pdms_points": float(np.abs(predicted - actual).mean()), "mean_actual_candidate_pdms": float(actual.mean()),
                "mean_pairwise_rms_dispersion_m": float(np.mean(dispersions)), "mean_waypoint_change_from_original_m": float(movement.mean()),
                "selected_nc_failure_count": int(np.sum(chosen_metrics[:, 0] < 1)), "selected_dac_failure_count": int(np.sum(chosen_metrics[:, 1] < 1)),
                "selected_ttc_failure_count": int(np.sum(chosen_metrics[:, 3] < 1)), "selected_comfort_failure_count": int(np.sum(chosen_metrics[:, 4] < 1)),
                "by_observed_command": {command: {"scene_count": int(sum(records[index]["command_name"] == command for index in indices)),
                    "selected_pdms": float(selected_pdms[[records[index]["command_name"] == command for index in indices]].mean())} for command in ["left_turn", "straight", "right_turn"]}}
            comparisons = {}
            if step == 300:
                for condition, source_index in [("A_original", 0), ("B_quality", 3), ("C_compatible", 4)]:
                    original_selected = source_predictions["log_scores"][source_index, indices].argmax(-1)
                    original_scores = source_metrics[source_index, indices, original_selected, -1] * 100
                    difference = selected_pdms - original_scores
                    comparisons["D_minus_" + condition] = {"selected_pdms_difference": float(difference.mean()),
                        "recording_bootstrap95ci": quantile_interval([difference[draw].mean() for draw in bootstrap])}
            row["paired_comparisons"] = comparisons; condition_reports.append(row)
            for offset, scene_index in enumerate(indices):
                record = records[scene_index]
                rows_for_csv.append({"token": record["token"], "recording_group": record["recording_group"], "subset": subset, "completed_steps": step,
                    "selected_pdms": float(selected_pdms[offset]), "oracle_best_pdms": float(oracle[offset]), "prediction_mae_pdms_points": float(np.abs(predicted[offset] - actual[offset]).mean()),
                    "selected_candidate_index": int(chosen[offset]), **{name: float(chosen_metrics[offset, metric_index]) for metric_index, name in enumerate(METRIC_NAMES[:-1])}})
    final_D = next(row for row in condition_reports if row["subset"] == "locked_development" and row["completed_steps"] == 300)
    summaries = [source_result["results"]["locked_development"]["conditions"][index] for index in [0, 3, 4]] + [final_D]
    result = {"complete": True, "condition": "D_accuracy_only", "training": training, "results": condition_reports,
        "A_B_C_D_common_development": summaries, "same_cache_seed_initialization_and_batch_order": True,
        "quality_terms_removed": ["target_actual_quality_weight", "target_quality_guard", "quality_coverage_loss"],
        "navtest_teacher_or_gradient_rows": 0, "full_navtest": False, "independent_final_test": False,
        "cost": {"training_cpu_seconds": training["cpu_wall_seconds"], "prediction_cpu_seconds": metadata["cpu_wall_seconds"],
            **json.loads((output / "evaluation_complete.json").read_text()), "new_gpu_seconds": 0},
        "protected_training": protected_files_audit(reference_configuration(source_configuration))}
    write_json(destination / "results.json", result)
    with (destination / "scene_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows_for_csv[0])); writer.writeheader(); writer.writerows(rows_for_csv)
    with (destination / "comparison.csv").open("w", newline="") as stream:
        fields = ["condition", "selected_pdms", "oracle_best_pdms", "selection_regret_pdms_points", "prediction_mae_pdms_points", "mean_pairwise_rms_dispersion_m", "mean_waypoint_change_from_original_m", "selected_nc_failure_count", "selected_dac_failure_count", "selected_ttc_failure_count"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows([{name: row[name] for name in fields} for row in summaries])
    render(configuration, summaries, training, output, destination)
    lines = ["# D — 예측오차만으로 목표 분포를 만드는 generator 추가 실험", "",
        f"## 결론\n\nD의후보예측MAE는원A17.4045→{final_D['prediction_mae_pdms_points']:.4f}로줄었으나실제선택PDMS는A93.6087→{final_D['selected_pdms']:.4f}였다. 현재accuracy-only목적의planning이득을입증하지못했다. 추가fullscale학습은시작하지않았다. 작은개발표본이므로모든유사방법의불가능성이나평균악화의통계적확정으로확대하지않는다.", "",
        "사용자 요청에 따라 실제 품질 가중·품질 guard·품질 coverage를 모두 제거했다. 원좌표 anchor만 유지했고 원래 head/cache/배치순서/seed0/300step/batch8/LR1e-5는 B/C와 같다. B/C에서 이어학습하지 않았다.", "",
        "```text", "q_D(i|scene) = softmax(-abs(predicted_PDMS_i - actual_PDMS_i) / 0.15)",
        "L_D = MMD²(uniform generated64, weighted native64 by q_D) + 0.1 × native-coordinate anchor", "```", "",
        "**작은 예측오차가 좋은 주행을 뜻하지는 않는다.** 실제0점 후보를0점으로맞혀도D는높은가중치를줄수있다. 실제정답은오차정의에만사용하며품질선호나안전teacher로는사용하지않는다. NAVTRAIN128에서만target/gradient를만들고NAVTEST는평가에만사용했다.", "",
        f"학습용Dteacher실제품질평균{training['training_target_statistics']['target_mean_actual_pdms']:.2f},예측오차평균{training['training_target_statistics']['target_mean_absolute_prediction_error_pdms_points']:.2f},실제0점후보targetmass{100*training['training_target_statistics']['target_mass_on_actual_pdms_zero']:.2f}%다. Teacher통계는inference성능이아니다.", "",
        "| 조건 | 선택 PDMS | 후보 최고 | 선택손실 | 후보 예측 MAE | NC/DAC/TTC 실패 수 |", "|---|---:|---:|---:|---:|---|"]
    for row in summaries:
        lines.append(f"| {row['condition']} | {row['selected_pdms']:.4f} | {row['oracle_best_pdms']:.4f} | {row['selection_regret_pdms_points']:.4f} | {row['prediction_mae_pdms_points']:.4f} | {row['selected_nc_failure_count']}/{row['selected_dac_failure_count']}/{row['selected_ttc_failure_count']} |")
    lines += ["", "위PDMS는동일NAVTEST64/38recording에서각scorerTop1선택경로를공식평가한평균이다. 후보점수평균이나전체NAVTEST결과가아니다. 동일publicscorer/full64context를사용하고변경된경로official라벨은새로계산했다.", "",
        f"D의대응차이와95%recording구간: {json.dumps(final_D['paired_comparisons'], ensure_ascii=False)}", "",
        "단일seed/128학습장면/반복노출개발패널이다. D는target품질항뿐아니라explicitqualitycoverage도OFF이므로C−D를어느한항만의순수효과로부르지않는다. 원본A/B/C수치·checkpoint·cache는변경하지않았다. 전체후보MAE감소와최종선택정확도는별도결과이며, 낮은품질targetmass가이번선택악화의원인이라는인과분리도실시하지않았다.", "",
        "[A/B/C/D 비교·D 학습곡선](accuracy_only_comparison_and_training.png). 현재조건에서예측오차를작게만드는것과좋은후보·선택경로를만드는것이함께개선됐는지각지표로판단한다.", "",
        "[D가 선호하는 학습 후보](accuracy_only_teacher_distribution.png): 좌측actual/prediction,우측D확률. actual품질로가중하지않는다.", "",
        f"비용: {json.dumps(result['cost'], ensure_ascii=False)}. GPU/encoder호출0,모든학습stepgradient유한. 원래LPWM등록{result['protected_training']['checked_files']}파일불일치0·학습대기열보존. fullscale자동확장없음.", "",
        "```bash", "bash scripts/run_scorer_compatible_accuracy_only.sh", "```", ""]
    (destination / "report.md").write_text("\n".join(lines))
    write_json(output / "complete.json", {"complete": True, "results": configuration["results_directory"], "new_gpu_seconds": 0})
    print(json.dumps({"D": final_D, "teacher": training["training_target_statistics"]}), flush=True)


def render(configuration, summaries, training, output, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(16, 5)); names = ["A original", "B quality", "C quality+accuracy", "D accuracy only"]
    positions = np.arange(4)
    axes[0].bar(positions-.16, [row["selected_pdms"] for row in summaries], width=.3, label="Selected actual PDMS", color="#4c78a8")
    axes[0].bar(positions+.16, [row["oracle_best_pdms"] for row in summaries], width=.3, label="Oracle best", color="#81b267")
    axes[0].set_ylim(0, 105); axes[0].set_xticks(positions, names, rotation=17); axes[0].legend(fontsize=8)
    axes[1].bar(positions, [row["prediction_mae_pdms_points"] for row in summaries], color=["#888888", "#4c78a8", "#d04467", "#935ba8"])
    axes[1].set_xticks(positions, names, rotation=17); axes[1].set_ylabel("Candidate prediction MAE (PDMS pts)")
    logs = json.loads((output / "training_log.json").read_text()); smoothed = np.convolve([row["loss"] for row in logs], np.full(16, 1/16), mode="valid")
    axes[2].plot(np.arange(16, len(logs)+1), smoothed, color="#935ba8"); axes[2].set_xlabel("Optimizer step"); axes[2].set_title("D total loss | 16-step mean")
    figure.suptitle("Accuracy-only generator objective: same initialization, data and budget\nActual top1 planning and score accuracy are measured separately", fontsize=12)
    figure.tight_layout(rect=(0, .02, 1, .90)); figure.savefig(destination / "accuracy_only_comparison_and_training.png", dpi=145); plt.close(figure)
    source_configuration, source_output, registration, cache = load_source(configuration)
    teacher = np.load(output / "accuracy_only_teacher_weights.npy"); train_records = [record for record in registration["records"] if record["pilot_split"] == "train"]
    labels = np.load(source_output / "native_metrics_all.npy"); figure, axes = plt.subplots(3, 2, figsize=(12, 11))
    for row_index, command in enumerate(range(3)):
        train_index = next(index for index, record in enumerate(train_records) if record["command_index"] == command); record = train_records[train_index]; row = record["pilot_row"]
        actual = labels[row, :, -1] * 100; predicted = np.exp(cache["native_log_scores"][row]) / 12 * 100
        axes[row_index, 0].plot(actual, label="Actual PDMS", color="#1a9850"); axes[row_index, 0].plot(predicted, label="Scorer estimate", color="#e98528"); axes[row_index, 0].set_ylim(0, 105); axes[row_index, 0].legend(fontsize=8)
        axes[row_index, 0].set_title(record["command_name"] + " | " + record["token"])
        axes[row_index, 1].plot(teacher[train_index], color="#935ba8"); axes[row_index, 1].set_ylabel("D teacher probability"); axes[row_index, 1].set_xlabel("Candidate index")
    figure.suptitle("TRAINING-only teacher: low prediction error is preferred regardless of actual quality", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, .94)); figure.savefig(destination / "accuracy_only_teacher_distribution.png", dpi=140); plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--configuration", type=Path, required=True); parser.add_argument("--phase", choices=["train", "predict", "labels", "report"], required=True)
    arguments = parser.parse_args(); configuration = json.loads(arguments.configuration.read_text())
    {"train": train, "predict": predict, "labels": labels, "report": report}[arguments.phase](configuration)
