"""Replay the fixed future-repeat-current test and visualize actual trajectories.

This is an explanation of an existing checkpoint diagnostic, with no optimizer,
encoder inference, oracle evaluation, training changes, or altered labels.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MONITOR = ROOT / "outputs/lpwm_drivor_planning_path_representation_monitor_v1"
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
BLUE, ORANGE, GREEN = "#2563eb", "#ea580c", "#16a34a"


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def replay(arguments):
    import torch
    import monitor_lpwm_drivor_planning_path_representations as diagnostic

    assert not arguments.output.exists() or not any(arguments.output.iterdir()), "Preserve completed outputs"
    diagnostic.check_registration()
    destination = MONITOR / f"update_{arguments.updates:06d}"
    completion = read_json(destination / "complete.json")
    checkpoint_path = ROOT / f"outputs/lpwm_drivor_particle_trends_every500_v1/checkpoints/update_{arguments.updates:06d}.pt"
    assert completion["complete"] and digest(checkpoint_path) == completion["checkpoint_sha256"]
    assert diagnostic.card_bytes() < 42_000_000_000, "Training has priority"
    arguments.output.mkdir(parents=True, exist_ok=True)
    write_json(arguments.output / "registration.json", {
        "source_sha256": digest(Path(__file__)), "checkpoint_sha256": completion["checkpoint_sha256"],
        "panel_sha256": digest(MONITOR / "panel.json"), "attributes_sha256": digest(destination / "particle_attributes.npy"),
        "intervention_metrics_sha256": digest(destination / "interventions.json"),
        "started_unix": time.time(), "completed_updates": arguments.updates,
        "model_updates": False, "encoder_reinference": False,
        "intervention": "Replace all 8 future states (foreground and broadcast background, all 4 cameras) with current state attributes; preserve current state, images, ego inputs, checkpoint, planner",
        "case_selection": "First intervention scene of each scene_type in original panel order, independent of result direction",
        "scope": "Existing fixed 12-scene training diagnostic; not navtest"})
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(4 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    device = torch.device("cuda")
    configuration = read_json(ROOT / "configs/lpwm_drivor_planning_path_lora/navsim_v1.json")
    torch.manual_seed(2)
    model = diagnostic.LPWMDrivoRPlanningPathLoRAModel(ROOT / configuration["public_checkpoint"]).to(device).eval()
    native_digest = model.frozen_native_digest()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert state["completed_updates"] == arguments.updates
    model.load_state_dict(state["model"], strict=True)
    assert native_digest == state["frozen_native_sha256"] == model.frozen_native_digest()
    del state
    panel = read_json(MONITOR / "panel.json")
    rows = read_json(destination / "interventions.json")
    original_metrics = {row["token"]: row for row in rows if row["variant"] == "baseline"}
    repeated_metrics = {row["token"]: row for row in rows if row["variant"] == "future_repeat_current"}
    images = np.load(MONITOR / "images.npy", mmap_mode="r")
    attributes = np.load(destination / "particle_attributes.npy", mmap_mode="r")
    selected_indices = [index for index, record in enumerate(panel["records"]) if record["token"] in original_metrics]
    assert len(selected_indices) == 12
    trajectories_original, trajectories_repeated, expert_trajectories, checks = [], [], [], []
    maximum_card_bytes = 0
    for index in selected_indices:
        maximum_card_bytes = max(maximum_card_bytes, diagnostic.card_bytes())
        assert maximum_card_bytes < 46_500_000_000
        record = panel["records"][index]
        cache = Path(record["cache_directory"])
        ego = np.array(np.load(cache / "ego.npy", mmap_mode="r")[record["cache_row"]])
        expert = np.array(np.load(cache / "trajectory.npy", mmap_mode="r")[record["cache_row"]])
        features = {"image": torch.from_numpy(np.array(images[index])).to(device).permute(0, 3, 1, 2)[None].float() / 255,
                    "ego_status": torch.from_numpy(ego).to(device)[None, None]}
        original = np.array(attributes[index])
        repeated = original.copy()
        repeated[:, 1:] = original[:, :1]
        assert np.array_equal(original[:, 0], repeated[:, 0])
        assert all(np.array_equal(repeated[:, step], original[:, 0]) for step in range(1, 9))
        for variant, values, expected, collection in (
            ("baseline", original, original_metrics[record["token"]], trajectories_original),
            ("future_repeat_current", repeated, repeated_metrics[record["token"]], trajectories_repeated)):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                memory = diagnostic.memory_from_attributes(model, values)
                prediction = diagnostic.planning_with_memory(model, features, memory)
                trajectory = prediction["trajectory"].float().cpu().numpy()[0]
            assert trajectory.shape == (8, 3) and np.isfinite(trajectory).all()
            ade = float(np.linalg.norm(trajectory[:, :2] - expert[:, :2], axis=-1).mean())
            difference = abs(ade - expected["ade_m"])
            assert difference < 1e-6, (record["token"], variant, difference)
            checks.append({"token": record["token"], "variant": variant, "ade_m": ade, "saved_diagnostic_ade_difference": difference})
            collection.append(trajectory)
        expert_trajectories.append(expert)
        print(f"REPLAY scene={index} checked=both", flush=True)
    assert native_digest == model.frozen_native_digest()
    diagnostic.check_registration()
    np.savez_compressed(arguments.output / "trajectories.npz", scene_indices=np.array(selected_indices),
        tokens=np.asarray([panel["records"][index]["token"] for index in selected_indices]),
        original=np.asarray(trajectories_original), repeated_current=np.asarray(trajectories_repeated), expert=np.asarray(expert_trajectories))
    write_json(arguments.output / "replay_checks.json", {"checks": checks, "maximum_saved_ade_difference": max(row["saved_diagnostic_ade_difference"] for row in checks),
        "maximum_card_used_bytes": maximum_card_bytes, "frozen_native_sha256": native_digest, "model_updates": False})
    del model
    torch.cuda.empty_cache()


def render(arguments):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from PIL import Image, ImageDraw, ImageFont

    font = FontProperties(fname=FONT_PATH)
    plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False})
    # Register this TTC explicitly; local font caches may predate Korean fonts.
    matplotlib.font_manager.fontManager.addfont(FONT_PATH)
    panel = read_json(MONITOR / "panel.json")
    saved = np.load(arguments.output / "trajectories.npz")
    indices = saved["scene_indices"].tolist()
    original_errors = np.linalg.norm(saved["original"][..., :2] - saved["expert"][..., :2], axis=-1)
    repeated_errors = np.linalg.norm(saved["repeated_current"][..., :2] - saved["expert"][..., :2], axis=-1)
    ade_original, ade_repeated = original_errors.mean(1), repeated_errors.mean(1)
    changes = ade_repeated - ade_original
    mean_change = float(changes.mean())
    reported = read_json(MONITOR / f"update_{arguments.updates:06d}/summary.json")["paired_interventions"]["future_repeat_current"]["ade_m_change"]
    assert abs(mean_change - reported["difference"]) < 1e-6

    # Diagram: actual operation on learned attribute vectors, not rendered future video.
    canvas = Image.new("RGB", (1700, 720), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    def write(position, message, size=26, color="#0f172a"):
        draw.text(position, message, font=ImageFont.truetype(FONT_PATH, size), fill=color)
    write((34, 22), "미래 정보를 이용하는지 확인하는 검사", 40)
    write((34, 87), "같은 장면 · 같은 주행 명령 · 같은 3,500-update 모델로 두 번 경로를 계산합니다.", 28)
    write((34, 151), "① 원래 입력", 28, BLUE)
    write((34, 322), "② 검사 입력", 28, ORANGE)
    for row, color, box_fill in ((0, BLUE, "#dbeafe"), (1, ORANGE, "#ffedd5")):
        top = 200 + row * 171
        for step in range(9):
            left = 36 + step * 161
            draw.rounded_rectangle((left, top, left + 147, top + 102), radius=12, fill=box_fill, outline=color, width=2)
            write((left + 10, top + 8), "지금" if step == 0 else f"{step * .5:g}초 뒤", 23, color)
            write((left + 10, top + 52), "현재 표현" if step == 0 else ("예측 표현" if row == 0 else "현재 복사"), 23)
        write((1490, top + 28), "→ Planner", 26, color)
    write((34, 512), "바꾼 것: 미래 8칸의 particle 위치·크기·외관·presence·배경 표현", 27)
    write((34, 557), "유지한 것: 현재 표현, 4개 카메라 이미지, ego 상태·명령, 모델 가중치", 27)
    write((34, 619), f"12장면 평균 정답 경로 오차: {ade_original.mean():.2f}m → {ade_repeated.mean():.2f}m  (+{mean_change:.2f}m)", 32)
    canvas.save(arguments.output / "future_branch_test_explained.png")

    scenarios = {"straight": "직진", "left_turn": "좌회전", "right_turn": "우회전"}
    selected_by_type = {}
    for row, index in enumerate(indices):
        selected_by_type.setdefault(panel["records"][index]["scene_type"], row)
    example_rows = [selected_by_type[name] for name in ("right_turn", "straight", "left_turn")]

    def draw_case(axes, row):
        index = indices[row]
        record = panel["records"][index]
        with Image.open(record["current_camera_paths"][0]) as image:
            axes[0].imshow(image.convert("RGB"))
        axes[0].axis("off")
        axes[0].set_title(f"장면 {index:02d} · {scenarios[record['scene_type']]}\n현재 전방 카메라 (참고 영상)", fontsize=14)
        for values, color, label, linestyle in (
            (saved["expert"][row], GREEN, "정답 경로 (사람의 실제 주행)", "-"),
            (saved["original"][row], BLUE, "예측한 미래 표현 사용", "-"),
            (saved["repeated_current"][row], ORANGE, "미래를 현재 표현으로 대체", "--")):
            points = np.vstack((np.zeros((1, 3)), values))
            axes[1].plot(-points[:, 1], points[:, 0], "o", color=color, markersize=4)
            axes[1].plot(-points[:, 1], points[:, 0], color=color, label=label, linestyle=linestyle, linewidth=2)
        axes[1].plot(0, 0, "k^", markersize=9)
        axes[1].set(xlabel="왼쪽 ← 자차 기준 좌우 (m) → 오른쪽", ylabel="전방 거리 (m)")
        axes[1].set_aspect("equal", adjustable="datalim")
        axes[1].grid(alpha=.25)
        axes[1].legend(fontsize=9, loc="best")
        axes[1].set_title("세 경로를 위에서 본 비교", fontsize=14)
        times = np.arange(1, 9) * .5
        axes[2].plot(times, original_errors[row], "o-", color=BLUE, label=f"원래: 평균 {ade_original[row]:.2f}m")
        axes[2].plot(times, repeated_errors[row], "o--", color=ORANGE, label=f"대체: 평균 {ade_repeated[row]:.2f}m")
        axes[2].set(xlabel="계획 시점 (초)", ylabel="같은 시점의 정답과 거리 (m)", ylim=(0, None))
        axes[2].grid(alpha=.25)
        axes[2].legend(fontsize=10)
        axes[2].set_title(f"정답 경로 오차 변화: {changes[row]:+.2f}m", fontsize=14)

    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True,
                                gridspec_kw={"width_ratios": [1.3, 1, 1]})
    draw_case(axes, selected_by_type["right_turn"])
    figure.suptitle("실제 사례: 모델의 미래 표현을 현재 표현으로 바꾸면 경로가 어떻게 달라지는가?", fontsize=18)
    figure.savefig(arguments.output / "actual_right_turn_case.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(3, 3, figsize=(16, 13), constrained_layout=True,
                                gridspec_kw={"width_ratios": [1.3, 1, 1]})
    for axis_row, row in zip(axes, example_rows):
        draw_case(axis_row, row)
    figure.suptitle("같은 검사에서 나온 세 사례 | 각 유형의 첫 장면을 선택, 결과 방향으로 고르지 않음", fontsize=18)
    figure.savefig(arguments.output / "three_actual_cases.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
    positions = np.arange(len(indices))
    axes[0].bar(positions - .18, ade_original, width=.36, color=BLUE, label="예측한 미래 표현 사용")
    axes[0].bar(positions + .18, ade_repeated, width=.36, color=ORANGE, label="미래를 현재 표현으로 대체")
    axes[0].set(xticks=positions, xticklabels=[f"{index:02d}" for index in indices], xlabel="고정된 12장면 번호", ylabel="정답 경로와 평균 거리 ADE (m)")
    axes[0].legend()
    axes[0].set_title(f"12장면 평균: {ade_original.mean():.2f}m → {ade_repeated.mean():.2f}m", fontsize=16)
    bars = axes[1].bar(positions, changes, color=[ORANGE if value >= 0 else BLUE for value in changes])
    axes[1].bar_label(bars, fmt="%+.2f", padding=3, fontsize=9)
    axes[1].axhline(0, color="#334155", linewidth=1)
    axes[1].axhline(mean_change, color="#334155", linestyle="--", label=f"평균 {mean_change:+.2f}m")
    axes[1].set(xticks=positions, xticklabels=[f"{index:02d}" for index in indices], xlabel="장면 번호", ylabel="대체 후 오차 − 원래 오차 (m)", ylim=(-.7, 2.15))
    axes[1].set_title(f"대체 후 악화 {int((changes > 0).sum())}장면 / 개선 {int((changes < 0).sum())}장면", fontsize=16)
    axes[1].legend()
    for axis in axes:
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    figure.suptitle("+0.48m는 12장면 평균의 정답 경로 오차 증가량입니다", fontsize=18)
    figure.savefig(arguments.output / "all12_ade_comparison.png", dpi=160)
    plt.close(figure)
    write_json(arguments.output / "visualization_report.json", {
        "completed_updates": arguments.updates, "scene_count": 12,
        "mean_original_ade_m": float(ade_original.mean()), "mean_repeated_current_ade_m": float(ade_repeated.mean()),
        "mean_ade_increase_m": mean_change, "worsened_scenes": int((changes > 0).sum()), "improved_scenes": int((changes < 0).sum()),
        "cases": [{"scene_index": index, "token": panel["records"][index]["token"], "scene_type": panel["records"][index]["scene_type"],
            "original_ade_m": float(ade_original[row]), "repeated_current_ade_m": float(ade_repeated[row]), "ade_change_m": float(changes[row])}
            for row, index in enumerate(indices)],
        "illustrative_scene_indices": [indices[row] for row in example_rows],
        "same_scene_same_weights": True, "training_changes": False,
        "trajectory_plot_coordinate_system": "Current ego frame; horizontal=-ego_y (screen right), vertical=ego_x (forward), meters",
        "current_front_photo_is_reference_only": True, "model_uses_all_four_cameras": True,
        "error_definition": "Mean of 8 Euclidean XY distances to the expert trajectory at matching 0.5s-to-4s timestamps; not distance between the two model trajectories",
        "interpretation": "Shows reliance on future-branch attributes, not necessarily accurate physical future prediction. Replacement can be out of distribution; fixed 12 training scenes."})
    print(json.dumps({"output": str(arguments.output), "mean_before": float(ade_original.mean()), "mean_after": float(ade_repeated.mean()),
        "mean_change": mean_change, "examples": [indices[row] for row in example_rows]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=3500)
    parser.add_argument("--output", type=Path, default=ROOT / "results/lpwm_drivor_planning_path_lora_v1/future_branch_intervention_explained_update3500")
    parser.add_argument("--render-only", action="store_true")
    arguments = parser.parse_args()
    arguments.output = arguments.output.resolve()
    if not arguments.render_only:
        replay(arguments)
    render(arguments)
