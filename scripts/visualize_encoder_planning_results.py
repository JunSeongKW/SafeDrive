"""Render saved encoder experiment trajectories and real observations on CPU.

No model inference, training, metric rescoring, or generated camera images.
Case selection deliberately covers the best, median, and worst ADE changes;
these cases are illustrative and are not a representative performance sample.
"""

import hashlib
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from matplotlib.lines import Line2D
from PIL import Image


WORKSPACE = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = WORKSPACE / "outputs/encoder_future_learning_v1/visualization"
SHARE_DIRECTORY = WORKSPACE / "results/encoder_future_learning_v1/visualization"
SEEDS = (29, 47, 83)
ORIGINAL_COLOR = "#2874ac"
TRAINED_COLOR = "#d95f32"
FUTURE_COLOR = "#8056ad"
GOOD_COLOR = "#26886c"
BAD_COLOR = "#ce595f"
NEUTRAL_COLOR = "#bbc3cb"


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure_style():
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 14, "axes.labelsize": 11,
        "figure.facecolor": "#f6f8fb", "axes.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.unicode_minus": False, "svg.fonttype": "path",
        "savefig.facecolor": "#f6f8fb",
    })


def save_figure(figure, stem):
    figure.savefig(OUTPUT_DIRECTORY / f"{stem}.png", dpi=165)
    figure.savefig(SHARE_DIRECTORY / f"{stem}.pdf")
    plt.close(figure)


def load_experiment():
    summary = read_json(WORKSPACE / "results/encoder_future_learning_v1/combined_summary.json")
    original_path = WORKSPACE / "outputs/encoder_future_learning_v1/original_baseline.json"
    original_rows = read_json(original_path)["windows"]
    tokens = [row["token"] for row in original_rows]
    assert len(tokens) == len(set(tokens)) == 192
    conditions = (
        "encoder_planning", "encoder_planning_last6", "intent_planning_last6",
        "intent_uniform_future_last6", "intent_uniform_future_last6_strong_auxiliary",
    )
    rows_by_condition = {"original_frozen": {seed: original_rows for seed in SEEDS}}
    summary_path = WORKSPACE / "results/encoder_future_learning_v1/combined_summary.json"
    sources = {str(path.relative_to(WORKSPACE)): sha256_file(path) for path in (original_path, summary_path)}
    for condition in conditions:
        family = "encoder_future_additional_controls_v1" if "strong_auxiliary" in condition else "encoder_future_learning_v1"
        rows_by_condition[condition] = {}
        for seed in SEEDS:
            path = WORKSPACE / "outputs" / family / f"{condition}_seed{seed}/results.json"
            report = read_json(path)
            assert report["completed_updates"] == 512
            indexed = {row["token"]: row for row in report["evaluations"]["512"]["development"]["windows"]}
            assert set(indexed) == set(tokens)
            rows_by_condition[condition][seed] = [indexed[token] for token in tokens]
            sources[str(path.relative_to(WORKSPACE))] = sha256_file(path)
    trajectories = {condition: np.asarray([[row["trajectory"] for row in seed_rows[seed]] for seed in SEEDS])
                    for condition, seed_rows in rows_by_condition.items()}
    ade = {condition: np.asarray([[row["xy_ade_m"] for row in seed_rows[seed]] for seed in SEEDS])
           for condition, seed_rows in rows_by_condition.items()}
    return summary, rows_by_condition, trajectories, ade, sources


def render_overview(summary, ade):
    figure, axes = plt.subplots(2, 2, figsize=(14, 10.3))
    figure.suptitle("Encoder 재학습 결과를 거리와 상황으로 보기", fontsize=22, x=.04, ha="left", y=.97)
    figure.text(.04, .92, "원본과 동일한 개발 192개 구간 · 3개 seed · 마지막 512 update 결과", fontsize=12, color="#536170")

    conditions = ["encoder_planning", "encoder_planning_last6", "intent_planning_last6",
                  "intent_uniform_future_last6", "intent_uniform_future_last6_strong_auxiliary"]
    labels = ["2블록 · planning", "6블록 · planning", "6블록 · intent + planning",
              "위 조건 + 미래 감독", "위 조건 + 강한 미래 감독"]
    axis = axes[0, 0]
    for position, condition in enumerate(conditions):
        comparison = summary["paired_ade_comparisons"][condition + "_minus_original_frozen"]
        interval = np.asarray(comparison["recording_cluster_bootstrap_95_ci_m"]) * 100
        color = FUTURE_COLOR if "future" in condition else TRAINED_COLOR
        axis.hlines(position, *interval, color=color, lw=3)
        axis.scatter(comparison["mean_difference_m"] * 100, position, c=color, s=65, zorder=3)
    axis.set_yticks(range(len(labels)), labels)
    axis.invert_yaxis()
    axis.axvline(0, color="#697786", ls="--", lw=1)
    axis.set_xlim(-1.23, .12)
    axis.set_title("① 평균 경로 오차 감소는 약 0.5–0.6 cm", loc="left", pad=16)
    axis.set_xlabel("원본 대비 ADE 변화 (cm)    ← 오차 감소")
    axis.grid(axis="x", alpha=.15)

    axis = axes[0, 1]
    counted_conditions = ["encoder_planning", "encoder_planning_last6", "intent_uniform_future_last6_strong_auxiliary"]
    counted_labels = ["2블록 planning", "6블록 planning", "6블록 intent + 강한 미래 감독"]
    counts = {}
    for position, condition in enumerate(counted_conditions):
        difference = (ade[condition] - ade["original_frozen"]).mean(axis=0)
        count = [int((difference < -.001).sum()), int((np.abs(difference) <= .001).sum()), int((difference > .001).sum())]
        assert sum(count) == 192
        counts[condition] = dict(zip(["improved_more_than_1mm", "within_1mm", "worse_more_than_1mm"], count))
        left = 0
        for size, color in zip(count, [GOOD_COLOR, NEUTRAL_COLOR, BAD_COLOR]):
            axis.barh(position, size, left=left, color=color, height=.58)
            if size:
                axis.text(left + size/2, position, str(size), ha="center", va="center", color="white" if color != NEUTRAL_COLOR else "#334150", fontsize=11)
            left += size
    axis.set_yticks(range(3), counted_labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 192)
    axis.set_xlabel("평가 구간 수 (192개)")
    axis.set_title("② 좋아진 구간과 나빠진 구간이 함께 존재", loc="left", pad=16)
    axis.legend([Line2D([0], [0], lw=8, color=color) for color in [GOOD_COLOR, NEUTRAL_COLOR, BAD_COLOR]],
                ["오차 감소", "±1 mm 이내", "오차 증가"], loc="upper center", bbox_to_anchor=(.5, -.22), ncol=3, frameon=False, fontsize=10)

    axis = axes[1, 0]
    change = (ade["encoder_planning_last6"] - ade["original_frozen"]).mean(axis=0) * 100
    order = np.argsort(change)
    axis.bar(np.arange(192), change[order], color=np.where(change[order] < 0, GOOD_COLOR, BAD_COLOR), width=1)
    axis.axhline(0, color="#697786", lw=1)
    axis.set_title("③ 6블록 planning의 장면별 변화", loc="left", pad=16)
    axis.set_xlabel("구간별 오차 변화 순서 (각 막대는 한 평가 구간)")
    axis.set_ylabel("원본 대비 ADE 변화 (cm)\n음수: 개선 / 양수: 악화")
    axis.set_xlim(-2, 193)
    axis.grid(axis="y", alpha=.15)

    axis = axes[1, 1]
    contexts = summary["pdm_context_breakdown"]["encoder_planning_last6"]["command"]
    changes = [contexts[command]["mean_difference_from_original"] * 100 for command in ["0", "1", "2"]]
    context_labels = [f"{label}\n{contexts[command]['unique_windows']}개" for command, label in [("0", "좌회전 명령"), ("1", "직진 명령"), ("2", "우회전 명령")]]
    axis.bar(range(3), changes, color=[GOOD_COLOR if value > 0 else BAD_COLOR for value in changes], width=.58)
    for position, value in enumerate(changes):
        axis.text(position, value + (.14 if value > 0 else -.18), f"{value:+.2f}", ha="center", va="bottom" if value > 0 else "top", fontsize=14)
    axis.axhline(0, color="#697786", lw=1)
    axis.set_xticks(range(3), context_labels)
    axis.set_ylim(-3.35, 3.3)
    axis.set_title("④ 평균 PDMS 상승이 모든 상황에 해당하진 않음", loc="left", pad=16)
    axis.set_ylabel("원본 대비 PDM 변화 (percentage point)")
    axis.grid(axis="y", alpha=.15)

    figure.subplots_adjust(left=.20, right=.97, top=.84, bottom=.13, hspace=.72, wspace=.55)
    figure.text(.04, .035,
                "① 선: seed 평균 차이의 recording bootstrap 95% 구간 (다중 비교 보정 없음). ②③: 구간별 ADE의 seed 평균 변화.\n"
                "②의 ±1 mm는 표시용 구간이며 통계적 동등성 기준이 아님. ④는 현재 route 명령별 기술 통계. 독립 test 결과가 아님.",
                fontsize=10, color="#536170")
    save_figure(figure, "01_encoder_learning_overview")
    return counts


def load_current_reference(record):
    log_path = WORKSPACE / "dataset/navsim_logs/trainval" / record["segment_filename"]
    with log_path.open("rb") as stream:
        current_frame = pickle.load(stream)[record["start_index"] + 3]
    assert current_frame["token"] == record["current_frame_token"]
    relative_path = next(camera["data_path"] for name, camera in current_frame["cams"].items() if name.lower() == "cam_f0")
    image_path = WORKSPACE / "dataset/sensor_blobs/trainval" / relative_path
    with Image.open(image_path) as reference:
        rgb = np.asarray(reference.convert("RGB"))
    return rgb, {"current_frame_token": record["current_frame_token"], "image_path": str(image_path), "image_sha256": sha256_file(image_path)}


def render_scene_cases(rows, trajectories, ade):
    configuration = read_json(WORKSPACE / "configs/encoder_future_learning/controlled_comparison_v1.json")
    records = read_json(WORKSPACE / configuration["reused_cache"])["records"]
    records_by_token = {record["current_frame_token"]: record for record in records}
    difference = ade["encoder_planning_last6"][0] - ade["original_frozen"][0]
    order = np.argsort(difference, kind="stable")
    positions = [int(order[0]), int(order[len(order)//2]), int(order[-1])]
    case_titles = ["오차 감소가 가장 큰 예시", "오차 변화가 중앙값에 가까운 예시", "오차 증가가 가장 큰 예시"]
    command_names = {0: "좌회전", 1: "직진", 2: "우회전"}
    time_points = np.arange(1, 9) * .5
    figure, axes = plt.subplots(3, 3, figsize=(16, 12), gridspec_kw={"width_ratios": [1.35, 1., 1.1]})
    figure.suptitle("실제 장면에서 원본과 재학습 경로는 어떻게 다른가", x=.04, y=.98, ha="left", fontsize=22)
    figure.text(.04, .945, "원본 vs 마지막 6블록 planning 학습 · seed 29 · 아래 세 장면은 개선 / 중앙 / 악화 사례를 의도적으로 선택", fontsize=12, color="#536170")
    audits = []
    for row_index, (position, case_title) in enumerate(zip(positions, case_titles)):
        row = rows["original_frozen"][29][position]
        record = records_by_token[row["token"]]
        original_trajectory = trajectories["original_frozen"][0, position, :, :2]
        trained_trajectory = trajectories["encoder_planning_last6"][0, position, :, :2]
        cache_digest = sha256_file(record["cache_file"])
        assert cache_digest == record["cache_sha256"]
        cached = torch.load(record["cache_file"], map_location="cpu", weights_only=True)
        target = cached["ego_trajectory_target"].numpy()[:, :2]
        original_error = np.linalg.norm(original_trajectory - target, axis=-1)
        trained_error = np.linalg.norm(trained_trajectory - target, axis=-1)
        assert np.isclose(original_error.mean(), row["xy_ade_m"], atol=1e-6)
        assert np.isclose(trained_error.mean(), ade["encoder_planning_last6"][0, position], atol=1e-6)
        reference_image, provenance = load_current_reference(record)
        camera_axis, trajectory_axis, error_axis = axes[row_index]
        camera_axis.imshow(reference_image)
        camera_axis.axis("off")
        camera_axis.set_title(f"{case_title}\n{command_names[row['command']]} 명령 · 현재 {row['ego_speed_meters_per_second']*3.6:.1f} km/h", loc="left", pad=10, fontsize=13)
        camera_axis.text(0, -.06, f"실제 현재 전방 영상  |  {row['token']}", transform=camera_axis.transAxes, fontsize=9, color="#536170")

        for trajectory, color, style, width in [(target, "#242c35", "-", 2.3), (original_trajectory, ORIGINAL_COLOR, "-", 2.7), (trained_trajectory, TRAINED_COLOR, "--", 2.0)]:
            with_origin = np.vstack((np.zeros((1, 2)), trajectory))
            trajectory_axis.plot(-with_origin[:, 1], with_origin[:, 0], color=color, ls=style, lw=width)
            trajectory_axis.scatter(-trajectory[-1, 1], trajectory[-1, 0], color=color, s=25, zorder=5)
        trajectory_axis.scatter(0, 0, color="#242c35", marker="^", s=70)
        trajectory_axis.set_aspect("equal", adjustable="datalim")
        trajectory_axis.set_xlabel("횡방향 (m): 왼쪽 ← → 오른쪽", fontsize=10)
        trajectory_axis.set_ylabel("전방 거리 (m)", fontsize=10)
        trajectory_axis.set_title("위에서 본 4초 예측 경로", fontsize=13)
        trajectory_axis.grid(alpha=.18)
        endpoint_difference = np.linalg.norm(trained_trajectory[-1] - original_trajectory[-1]) * 100
        trajectory_axis.text(.02, .98, f"4초 끝점 이동: {endpoint_difference:.1f} cm", transform=trajectory_axis.transAxes,
                             va="top", fontsize=10, bbox={"facecolor":"white", "edgecolor":"none", "alpha":.85})

        error_axis.plot(time_points, original_error * 100, color=ORIGINAL_COLOR, marker="o", lw=2, label="원본")
        error_axis.plot(time_points, trained_error * 100, color=TRAINED_COLOR, marker="o", ls="--", lw=2, label="6블록 학습")
        error_axis.fill_between(time_points, original_error * 100, trained_error * 100, color=GOOD_COLOR if difference[position] < 0 else BAD_COLOR, alpha=.14)
        error_axis.set_title(f"GT 대비 평균 오차: {original_error.mean()*100:.1f} → {trained_error.mean()*100:.1f} cm", fontsize=13)
        error_axis.set_xlabel("예측 시점 (초)")
        error_axis.set_ylabel("같은 시점의 GT와 거리 (cm)")
        error_axis.set_xticks([1, 2, 3, 4])
        error_axis.set_ylim(bottom=0)
        error_axis.grid(alpha=.18)
        error_axis.text(.03, .97, f"ADE 변화 {difference[position]*100:+.2f} cm", transform=error_axis.transAxes, va="top",
                        color=GOOD_COLOR if difference[position] < 0 else BAD_COLOR, fontsize=12, bbox={"facecolor":"white", "edgecolor":"none", "alpha":.85})
        audits.append({"selection":case_title, "token":row["token"], "seed":29,
                       "ade_change_m": float(difference[position]), "endpoint_change_cm":float(endpoint_difference),
                       "original_ade_m":float(original_error.mean()), "trained_ade_m":float(trained_error.mean()),
                       "cached_ground_truth_matches_reported_ade": True,
                       "ground_truth_cache_sha256":cache_digest, **provenance})
    figure.legend([Line2D([0],[0], color="#242c35", lw=2), Line2D([0],[0], color=ORIGINAL_COLOR, lw=2), Line2D([0],[0], color=TRAINED_COLOR, lw=2, ls="--")],
                  ["기록된 실제 ego 경로 (GT)", "원본 모델", "6블록 planning 학습"], loc="lower center", bbox_to_anchor=(.52,.038), ncol=3, frameon=False, fontsize=12)
    figure.subplots_adjust(left=.04, right=.97, top=.89, bottom=.11, hspace=.46, wspace=.38)
    figure.text(.04,.012,"사진은 실제 관측이며 생성 이미지가 아님. BEV는 현재 ego 좌표계의 예측 궤적이며 도로 지도·closed-loop 재생이 아님. 사례 선택은 전체 성능의 대표 표본이 아님.", fontsize=10, color="#536170")
    save_figure(figure, "02_observed_scenes_and_trajectories")
    return audits


def render_future_effect(trajectories, ade):
    baseline_change = np.linalg.norm(trajectories["encoder_planning_last6"][..., :2] - trajectories["original_frozen"][..., :2], axis=-1).reshape(-1, 8)
    future_change = np.linalg.norm(trajectories["intent_uniform_future_last6_strong_auxiliary"][..., :2] - trajectories["intent_planning_last6"][..., :2], axis=-1).reshape(-1, 8)
    future_ade_change = (ade["intent_uniform_future_last6_strong_auxiliary"] - ade["intent_planning_last6"]).mean(axis=0) * 1000
    time_points = np.arange(1, 9) * .5
    figure, axes = plt.subplots(1, 3, figsize=(15, 6.2))
    figure.suptitle("미래 감독을 추가했을 때의 변화는 어느 정도인가", x=.04, y=.96, ha="left", fontsize=22)
    figure.text(.04,.88,"미래 감독은 동일한 6블록 + intent planning 대조군과 비교 · 미래 loss 가중치를 10배 높인 조건", fontsize=12, color="#536170")
    for values, color, label in [(baseline_change, TRAINED_COLOR, "원본 → 6블록 planning"), (future_change, FUTURE_COLOR, "6블록 intent → 미래 감독 추가")]:
        axes[0].plot(time_points, np.median(values,axis=0)*100, color=color, lw=2.5, marker="o", label=label)
    axes[0].set_title("① 같은 cm 축에서 경로 이동량 비교", loc="left", pad=15)
    axes[0].set_ylabel("두 모델의 예측 위치 간 거리 중앙값 (cm)")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    axes[0].set_ylim(bottom=0)

    median = np.median(future_change,axis=0)*1000
    lower, upper = np.quantile(future_change,[.1,.9],axis=0)*1000
    axes[1].plot(time_points, median, color=FUTURE_COLOR, lw=2.5, marker="o")
    axes[1].fill_between(time_points,lower,upper,color=FUTURE_COLOR,alpha=.17)
    axes[1].set_title("② 미래 감독 추가분만 mm로 확대", loc="left", pad=15)
    axes[1].set_ylabel("동일 시점 예측 위치 이동량 (mm)")
    axes[1].text(.04,.97,"선: 중앙값\n영역: 10–90백분위\n(192구간 × 3 seed)",transform=axes[1].transAxes,va="top",fontsize=10)
    axes[1].set_ylim(bottom=0)
    for axis in axes[:2]:
        axis.set_xlabel("예측 시점 (초)")
        axis.set_xticks([1,2,3,4])
        axis.grid(alpha=.18)

    order=np.argsort(future_ade_change)
    axes[2].bar(np.arange(192),future_ade_change[order],color=np.where(future_ade_change[order] < 0,GOOD_COLOR,BAD_COLOR),width=1)
    axes[2].axhline(0,color="#697786",lw=1)
    axes[2].set_title("③ 이동했다고 오차가 줄어든 것은 아님",loc="left",pad=15)
    axes[2].set_xlabel("구간별 변화 순서 (각 막대는 한 평가 구간)")
    axes[2].set_ylabel("미래 감독 추가에 따른 ADE 변화 (mm)\n음수: 개선 / 양수: 악화")
    axes[2].grid(axis="y",alpha=.18)
    axes[2].set_xlim(-2,193)
    figure.subplots_adjust(left=.07,right=.975,top=.74,bottom=.23,wspace=.4)
    figure.text(.04,.065,"위치 이동량은 정답과의 오차가 아니라 두 모델 출력의 차이. ①②는 위치별 차이의 기술 통계이며 신뢰구간이 아님.\n"
                "③은 각 구간 ADE 변화의 seed 평균. 전체 scene-macro ADE 변화는 약 +0.007 mm로, 이번 미래 감독의 실질적 추가 이득은 확인하지 못함.",fontsize=10,color="#536170")
    save_figure(figure,"03_future_supervision_effect")
    return {"planning_change_median_cm_by_time":(np.median(baseline_change,axis=0)*100).tolist(),
            "future_change_median_mm_by_time":median.tolist(),
            "future_change_p10_mm_by_time":lower.tolist(),"future_change_p90_mm_by_time":upper.tolist(),
            "future_ade_change_mm_quantiles":np.quantile(future_ade_change,[0,.1,.5,.9,1]).tolist()}


def main():
    torch.set_num_threads(1)
    configure_style()
    OUTPUT_DIRECTORY.mkdir(parents=True,exist_ok=True)
    SHARE_DIRECTORY.mkdir(parents=True,exist_ok=True)
    summary, rows, trajectories, ade, sources = load_experiment()
    counts = render_overview(summary,ade)
    cases = render_scene_cases(rows,trajectories,ade)
    future_effect = render_future_effect(trajectories,ade)
    for path,digest in sources.items():
        assert sha256_file(WORKSPACE/path) == digest
    audit = {"complete":True,"source_results_unchanged":True,"new_model_inference":False,"new_training":False,
             "gpu_used":False,"scope":"Saved development results; 192 windows,182 unique scenes,24 recordings,3 seeds. Not independent test.",
             "sources":sources,"window_change_counts":counts,"scene_cases":cases,"future_effect":future_effect,
             "case_selection":"Seed29 planning-last6 vs original ADE change: minimum, upper median, maximum; illustrative outcome-selected cases, not representative sampling.",
             "plot_script_sha256":sha256_file(Path(__file__)),
             "images":[str(path.relative_to(WORKSPACE)) for path in sorted(OUTPUT_DIRECTORY.glob('*.png'))]}
    (SHARE_DIRECTORY/"visualization_audit.json").write_text(json.dumps(audit,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"counts":counts,"cases":cases,"future_effect":future_effect},indent=2,ensure_ascii=False))
    print("ENCODER_RESULTS_VISUALIZATION_COMPLETE")


if __name__ == "__main__":
    main()
