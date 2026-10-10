"""Paired normalization-only comparison and same-model intervention plots."""
import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from run_soft_token_reweighting import (
    PROJECT_ROOT, EncoderFeatureCache, model_from_initial, paired_recording_interval,
    prediction_from_minibatch, read_json, setup_official, sha256, write_json,
)


def score_array(path, records):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["token"] for row in rows] == [row["token"] for row in records]
    return np.array([float(row["score"])*100 for row in rows])


def make_intervention_figures(configuration, output, previous, result_directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    setup_official()
    torch.set_num_threads(2)
    cache = EncoderFeatureCache(output)
    validation_indices = cache.indices("validation")
    commands = cache.arrays["ego_status"][validation_indices, :4].argmax(-1)
    selected_offsets = [int(np.flatnonzero(commands == command)[0]) for command in (0, 1, 2)]
    selected_indices = validation_indices[selected_offsets]
    observations = cache.minibatch(selected_indices, "cpu")
    model = model_from_initial(output, "conditioned").eval()
    model.load_state_dict(torch.load(output / "conditioned/checkpoint.pt", map_location="cpu")["model"], strict=True)
    beta = float(model.importance.beta)
    with torch.no_grad():
        normal_prediction = prediction_from_minibatch(model, observations, capture_attention=True)
        normal_attention = model.last_attention.mean((1, 2))[:, :128].numpy().copy()
        normalized_bias = (beta*model.last_importance).numpy().copy()
        zero_prediction = prediction_from_minibatch(model, observations, intervention="beta_zero", capture_attention=True)
        zero_attention = model.last_attention.mean((1, 2))[:, :128].numpy().copy()
        command_biases, command_attention, command_predictions = [], [], []
        for command in (0, 1, 2):
            # Only the importance MLP's ego embedding receives this alternative.
            # The native planner's ego status remains the real observed status.
            def replace_importance_command(module, inputs, command=command):
                altered = inputs[0].clone()
                altered[:, :4] = 0
                altered[:, command] = 1
                return (altered,)
            handle = model.importance.ego_embedding.register_forward_pre_hook(replace_importance_command)
            command_predictions.append(prediction_from_minibatch(model, observations, capture_attention=True).numpy())
            command_biases.append((beta*model.last_importance).numpy().copy())
            command_attention.append(model.last_attention.mean((1, 2))[:, :128].numpy().copy())
            handle.remove()
    old_beta = read_json(previous / "conditioned/training_complete.json")["beta"]
    old_bias = old_beta*np.load(previous / "conditioned/token_diagnostics.npz")["importance"][selected_offsets]
    attention_delta = normal_attention-zero_attention
    signed_limit = max(float(np.abs(normalized_bias).max()), float(np.abs(old_bias).max()), 1e-12)
    delta_limit = max(float(np.abs(attention_delta).max()), 1e-12)
    attention_limit = max(float(normal_attention.max()), float(zero_attention.max()), 1e-12)
    figure, axes = plt.subplots(3, 6, figsize=(22, 10), squeeze=False)
    mappings = {}
    for row_index, cache_index in enumerate(selected_indices):
        record = cache.records[cache_index]
        original_image = np.asarray(Image.open(record["observed_front_paths"][-1]).convert("RGB"))
        height, width = original_image.shape[:2]
        panels = [(None, "Input", None), (old_bias[row_index], "Previous C: bias", "bias"),
                  (normalized_bias[row_index], "Normalized C: bias", "bias"),
                  (zero_attention[row_index], "Same C: bias OFF", "attention"),
                  (normal_attention[row_index], "Same C: bias ON", "attention"),
                  (attention_delta[row_index], "Same C: ON minus OFF", "difference")]
        for column, (values, title, kind) in enumerate(panels):
            axis = axes[row_index, column]
            axis.imshow(original_image)
            if values is not None:
                limit = {"bias": signed_limit, "attention": attention_limit, "difference": delta_limit}[kind]
                mappings[kind] = axis.imshow(values.reshape(8, 16), extent=(0, width, height-28, 28),
                    origin="upper", interpolation="nearest", alpha=.6, aspect="auto",
                    cmap="magma" if kind == "attention" else "coolwarm",
                    vmin=0 if kind == "attention" else -limit, vmax=limit)
            axis.set(xlim=(0, width), ylim=(height, 0))
            axis.axis("off")
            suffix = ""
            if kind == "bias":
                suffix = f"; std={values.std():.4g}"
            axis.set_title(title+"\n"+["left", "straight", "right"][row_index]+suffix, fontsize=10)
    figure.suptitle("Normalization-only pilot | fixed validation examples\n"
        "Bias = beta*r in logits. Attention averages 8 heads x 8 waypoint queries; ego memory omitted.", fontsize=14)
    figure.tight_layout(rect=(0, .12, 1, .93))
    for kind, left, label in (("bias", .04, "Signed bias (same scale before/after)"),
                              ("attention", .37, "Mean attention probability"),
                              ("difference", .70, "Attention ON - OFF (signed)")):
        color_axis = figure.add_axes([left, .05, .25, .02])
        figure.colorbar(mappings[kind], cax=color_axis, orientation="horizontal").set_label(label)
    figure.savefig(result_directory / "normalized_bias_and_attention_difference.png", dpi=145)
    plt.close(figure)

    command_biases = np.stack(command_biases, axis=1)
    command_attention = np.stack(command_attention, axis=1)
    command_predictions = np.stack(command_predictions, axis=1)
    command_differences = command_biases-command_biases[:, 1:2]
    command_limit = max(float(np.abs(command_differences).max()), 1e-12)
    figure, axes = plt.subplots(3, 4, figsize=(15, 10), squeeze=False)
    for row_index, cache_index in enumerate(selected_indices):
        record = cache.records[cache_index]
        original_image = np.asarray(Image.open(record["observed_front_paths"][-1]).convert("RGB"))
        height, width = original_image.shape[:2]
        for column in range(4):
            axis = axes[row_index, column]
            axis.imshow(original_image)
            if column:
                mapping = axis.imshow(command_differences[row_index, column-1].reshape(8,16),
                    extent=(0,width,height-28,28), origin="upper", interpolation="nearest", alpha=.6,
                    cmap="coolwarm", vmin=-command_limit, vmax=command_limit, aspect="auto")
            axis.set(xlim=(0,width), ylim=(height,0)); axis.axis("off")
            axis.set_title("Input: "+["left", "straight", "right"][row_index] if column == 0 else
                ["left", "straight", "right"][column-1]+" importance command\nbias minus straight command")
    figure.suptitle("Same C, same scene: change the importance command only\n"
        "Native planner ego stays fixed. Straight reference is zero by construction; counterfactuals may be inconsistent with scene.")
    figure.tight_layout(rect=(0,.10,1,.92))
    color_axis = figure.add_axes([.25,.04,.5,.02])
    figure.colorbar(mapping,cax=color_axis,orientation="horizontal").set_label("Signed command-only bias difference (logits)")
    figure.savefig(result_directory / "same_scene_command_bias_difference.png",dpi=145)
    plt.close(figure)
    measurements = {"selection": "same first validation scene per command as v1, not selected for effect",
        "tokens": [cache.records[index]["token"] for index in selected_indices],
        "bias_shared_color_limit": signed_limit, "attention_difference_color_limit": delta_limit,
        "command_difference_color_limit": command_limit,
        "mean_absolute_attention_on_off_difference": float(np.abs(attention_delta).mean()),
        "maximum_absolute_attention_on_off_difference": float(np.abs(attention_delta).max()),
        "mean_waypoint_on_off_change_meters": float((normal_prediction[..., :2]-zero_prediction[..., :2]).norm(dim=-1).mean()),
        "maximum_command_bias_change_from_straight": float(np.abs(command_differences).max()),
        "mean_command_attention_change_from_straight": float(np.abs(command_attention-command_attention[:,1:2]).mean()),
        "mean_command_waypoint_change_from_straight_meters": float(np.linalg.norm(command_predictions[...,:2]-command_predictions[:,1:2,...,:2],axis=-1).mean()),
        "scope": "three fixed validation scenes only; grid anchors are not detected objects; not a planning-benefit test"}
    write_json(result_directory / "intervention_visualization.json", measurements)
    return measurements


def run(arguments):
    started = time.perf_counter()
    configuration = read_json(arguments.config)
    output = PROJECT_ROOT / configuration["output_directory"]
    previous = PROJECT_ROOT / configuration["reuse_inputs_from"]
    result_directory = PROJECT_ROOT / configuration["results_directory"]
    current_results = read_json(result_directory / "results.json")
    previous_results = read_json(PROJECT_ROOT / "results/soft_token_reweighting_v1/results.json")
    records = [row for row in read_json(output / "subset_manifest.json")["records"] if row["pilot_split"] == "validation"]
    assert sha256(output / "subset_manifest.json") == sha256(previous / "subset_manifest.json")
    assert sha256(output / "initial_planner.pt") == sha256(previous / "initial_planner.pt")
    assert sha256(output / "training_schedule.npy") == sha256(previous / "training_schedule.npy")
    comparisons, rows = {}, []
    for condition in configuration["conditions"]:
        current = current_results["models"][condition]
        old = previous_results["models"][condition]
        for key in ("steps", "batch_size", "seed", "initial_native_planner_state_sha256",
                    "schedule_sha256", "trainable_parameters", "native_planner_trainable_parameters"):
            assert current["training"][key] == old["training"][key], (condition,key)
        scores_current = score_array(output / condition / "scores_normal.csv", records)
        scores_previous = score_array(previous / condition / "scores_normal.csv", records)
        comparison = paired_recording_interval(scores_current-scores_previous, records)
        comparisons[condition] = comparison
        current_metrics = current["evaluation"]["results"]["normal"]
        old_metrics = old["evaluation"]["results"]["normal"]
        rows.append({"condition": condition,"previous_pdms": old_metrics["pdms"],
                     "normalized_pdms": current_metrics["pdms"],
                     "difference_points": comparison["mean_difference_points"],
                     "ade_meters": current_metrics["ade_meters"],"fde_meters": current_metrics["fde_meters"]})
    with (result_directory / "normalization_comparison.csv").open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    baseline_old=np.load(previous / "baseline/predictions_normal.npz")["trajectories"]
    baseline_current=np.load(output / "baseline/predictions_normal.npz")["trajectories"]
    baseline_difference=float(np.abs(baseline_old-baseline_current).max())
    previous_losses=[json.loads(line)["loss"] for line in (previous / "baseline/training.jsonl").read_text().splitlines()]
    current_losses=[json.loads(line)["loss"] for line in (output / "baseline/training.jsonl").read_text().splitlines()]
    first_different=next((index for index,(old,current) in enumerate(zip(previous_losses,current_losses)) if old != current),None)
    replay_audit={"first_different_training_loss_step":None if first_different is None else first_different+1,
        "first_loss_difference":0.0 if first_different is None else current_losses[first_different]-previous_losses[first_different],
        "strict_deterministic_algorithms_configured":False,
        "interpretation":"Identical seed/initial weights/schedule do not yield bitwise replay. Tiny early floating-point differences amplify; exact kernel cause was not isolated. Between-run score differences include replay variation."}
    write_json(result_directory / "baseline_replay_audit.json",replay_audit)
    figures = make_intervention_figures(configuration, output, previous, result_directory)
    reuse=read_json(output / "input_reuse.json")
    shutil.copy2(output / "input_reuse.json",result_directory / "input_reuse.json")
    shutil.copy2(output / "configuration.json",result_directory / "configuration.json")
    old_diagnostic=previous_results["bias_scale_and_ego_sensitivity"]
    current_diagnostic=current_results["bias_scale_and_ego_sensitivity"]
    result={"only_scientific_change": "valid-token centered importance / (population std + 1e-6)",
        "rows": rows,"paired_previous_comparisons":comparisons,
        "within_normalized_comparisons":current_results["comparisons"],
        "baseline_replay_prediction_max_absolute_difference":baseline_difference,
        "baseline_replay_audit":replay_audit,
        "same_manifest_initial_planner_schedule":True,
        "previous_mechanism":old_diagnostic,"normalized_mechanism":current_diagnostic,
        "visualization":figures,"cache_reuse":reuse,
        "training_gpu_hours":current_results["training_total_gpu_hours"],
        "single_seed":True,"validation_scenes":len(records),"comparison_seconds":time.perf_counter()-started}
    write_json(result_directory / "normalization_comparison.json",result)
    lines=["# 중요도 출력 정규화만 추가한 비교 실험", "",
        "연구 질문: 장면·ego 조건부 중요도의 작은 bias를 정규화로 키우면 실제 attention과 planning에 기여하는가?",
        "", "## 변경과 통제", "",
        "B/C 중요도 출력에 valid-token population 표준편차 정규화(분모 std+1e-6)만 추가했다. 초기 beta=0.1, MLP 및 출력층 초기화는 동일하다. 추가 파라미터·loss·학습률 변경은 없다. 정규화로 비균일한 중요도가 만들어지는 것 자체는 학습 효과나 객체 이해의 증거가 아니다.",
        "navtrain10,000 / navval1,000, seed0, batch32, 각1,000update, 동일 초기 planner·캐시·batch schedule의 SHA 일치를 확인했다. A도 재학습했으며 기존 A 대비 예측 최대 차이는 "+f"{baseline_difference:.9g}이다.",
        "", "## 실제 결과", "", "| 조건 | 이전 PDMS | 정규화 실험 PDMS | 이전 대비 | ADE m | FDE m |", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['condition']} | {row['previous_pdms']:.4f} | {row['normalized_pdms']:.4f} | {row['difference_points']:+.4f} | {row['ade_meters']:.4f} | {row['fde_meters']:.4f} |")
    lines += ["", "**재실행 변동:** 변경 없는 A도 이전 대비 "+f"{rows[0]['difference_points']:+.4f}점 달라졌다. 첫 loss 차이는 update{replay_audit['first_different_training_loss_step']}에서 {replay_audit['first_loss_difference']:.9g}였으며 이후 증폭됐다. seed·초기 상태·배치 순서와 초기 검증 예측은 동일하다. 엄격한 연산 결정성은 설정되지 않았고 정확한 kernel 원인은 분리하지 못했다. 따라서 이전→이번 점수 차이를 정규화만의 효과로 귀속할 수 없다. 이번 A/B/C를 주 비교로 쓰되 아래 구간도 학습 재실행/seed 변동은 반영하지 않는다."]
    lines += ["", "| 이번 실험 내 비교 | PDMS 차이 | 기록 단위 bootstrap 95% 구간 |", "|---|---:|---|"]
    for label,measurement in current_results["comparisons"].items():
        lines.append(f"| {label} | {measurement['mean_difference_points']:+.6f} | {measurement['recording_bootstrap_95_interval']} |")
    lines += ["", "## 기능적 사용과 성능을 구분", "",
        f"C의 추가 bias 표준편차: 이전 {old_diagnostic['mean_learned_bias_std']:.8f} → {current_diagnostic['mean_learned_bias_std']:.8f}. 기존 QK 대비 표준편차 비율 평균: {old_diagnostic['mean_bias_to_content_std_ratio']*100:.5f}% → {current_diagnostic['mean_bias_to_content_std_ratio']*100:.5f}%.",
        f"C에서 bias를 끌 때 평균 waypoint 변화: 이전 {old_diagnostic['trajectory_sensitivity']['beta_zero']['mean_waypoint_change_meters']:.9f}m → {current_diagnostic['trajectory_sensitivity']['beta_zero']['mean_waypoint_change_meters']:.9f}m.",
        "C의 정상·bias0·shuffle PDMS는 위 표로 판단한다. 기존 planner가 이미 ego를 받으므로 C가 ego를 쓰도록 보장되지는 않는다. 한 모델의 개입은 기능적 의존성 검사이며, 재학습 대조나 객체의 인과적 중요도 검증과 다르다."]
    lines += [f"이번 C의 명령 변경에 대한 중요도 변화는 장면 내 중요도 std의 약 {current_results['models']['conditioned']['prediction']['diagnostics']['command_only_counterfactual_relative_to_importance_std']*100:.3f}%다. 강한 주행 명령 의존을 배웠다는 근거는 없다.",
        "이번 bias OFF/shuffle의 점수 차이가 작다면, bias를 실제로 계산하고 attention을 바꾼다는 사실과 planning에서 유의미하게 활용한다는 주장은 구분해야 한다. 나머지 decoder 경로가 첫 층 bias의 영향을 약화할 가능성은 추정이며 이 실험으로 원인을 확정하지 않는다."]
    gain=current_results["comparisons"]["C_minus_A"]
    lower,upper=gain["recording_bootstrap_95_interval"]
    if lower>0:
        lines += ["이번 단일 seed 개발셋에서는 C가 A보다 높고 기록 단위 구간이0위다. 학습 seed 불확실성은 포함하지 않는다. 다음 실험 하나는 같은 조건의 추가 seed 반복이다."]
    elif upper<0:
        lines += ["이번 설정에서는 C가 A보다 낮다. bias 크기를 늘렸다는 사실을 planning 개선으로 해석할 수 없다. 다음 실험 하나는 동일 데이터에서 planner를 고정한 채 중요도 모듈만 학습하는 진단이다(이번에 실행하지 않음)."]
    else:
        lines += ["C−A의 기록 단위 구간에0이 포함된다. planning 개선은 불확실하다. 다음 실험 하나는 연산 재현성 설정을 먼저 검증한 뒤 동일 A/B/C를 추가 seed에서 반복하는 것이다(이번에 실행하지 않음)."]
    lines += ["", "## 비용·검증·한계", "",
        f"학습 합계 {current_results['training_total_gpu_hours']:.6f} GPU h. 기존 실제 feature 캐시를 재사용하여 이번 encoder/cache 생성0초; 재사용 검증 {reuse['reuse_verification_seconds']:.2f}초. profile/평가/CPU 시각화 비용은 학습 비용에 포함하지 않는다.",
        "기존6종 검증과 실제 training sample의 정규화 std·bias ON/OFF attention/경로 영향·단일 valid-token finite gradient 검사를 통과했다. CPU 검증 상세는 verification.json. 상황별 점수·하위 지표·파라미터·메모리·latency·gradient는 report.md/results.json에 있다.",
        "단일 seed, 과거 사용된 부분 navval 개발셋이다. 독립 test나 전체 NAVTEST 점수, 객체 이해/계산량 절감으로 해석하지 않는다. Bootstrap은 기록 간 변동만 다루며 학습 seed 변동은 다루지 않는다. beta는 자유롭게 학습되므로 정규화 후에도 필요하면 bias를 줄일 수 있다.",
        "", "## 시각화", "", "![Bias and attention intervention](normalized_bias_and_attention_difference.png)",
        "", "이전 C와 정규화 C의 bias는 같은 색상 범위다. 마지막 세 열은 동일한 정규화 C에서 OFF/ON/차이를 비교한다. Grid는 공간 anchor이며 객체 detection/segmentation이 아니다.",
        "", "![Same scene command](same_scene_command_bias_difference.png)",
        "", "동일 scene·동일 C에서 중요도 입력의 명령만 변경했다. native planner ego는 그대로다. 각 패널은 직진 명령 대비 bias 차이다. 영상과 일치하지 않는 명령도 있어 성능/인과 검증으로 해석하지 않는다.",
        "", "## 재실행", "", "```bash", "cd /rhome/junseong/PlanningAwareFuturePrediction",
        "/rhome/junseong/envs/kjs-drive-jepa-extension/bin/python scripts/run_soft_token_reweighting_suite.py --config configs/soft_token_reweighting/pilot_normalized_v2.json --replay-id rerun_001", "```"]
    (result_directory / "normalization_comparison.md").write_text("\n".join(lines)+"\n")
    print({"complete":True,"rows":rows,"results":str(result_directory)},flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    run(parser.parse_args())
