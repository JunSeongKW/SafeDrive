"""Complete LPWM development/navtest inference and unchanged official CPU PDM scoring.

Run prepare-navtest/score in the official NAVSIM environment, predict/summarize in
the LPWM environment. Independent navtest labels are used only for evaluation.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".pending.json")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def prepare_navtest(specification):
    import cv2
    from PIL import Image
    from audit_official_drive_jepa_evaluation import official_configuration, create_scene_loader
    from prepare_lpwm_full_planning_targets import ego_supervision
    output_root = PROJECT_ROOT / specification["output_directory"] / "navtest_inputs"
    output_root.mkdir(parents=True, exist_ok=True)
    if (output_root / "manifest.json").exists():
        return
    _, _, configuration, _ = official_configuration(PROJECT_ROOT)
    from navsim.common.dataloader import MetricCacheLoader
    scene_loader = create_scene_loader(configuration)
    metric_loader = MetricCacheLoader(Path(configuration.metric_cache_path))
    tokens = sorted(scene_loader.tokens)
    assert len(tokens) == 12146 and set(tokens) == set(configuration.train_test_split.scene_filter.tokens)
    records, statuses, trajectories = [], [], []
    for token in tokens:
        frames = scene_loader.scene_frames_dicts[token]
        assert len(frames) >= 12 and frames[3]["token"] == token
        images = [str(Path(configuration.sensor_blobs_path) / next(camera for name, camera in frame["cams"].items()
            if name.lower() == "cam_f0")["data_path"]) for frame in frames[:4]]
        assert all(Path(path).is_file() for path in images), f"Missing observed RGB: {token}"
        status, target = ego_supervision(frames)
        statuses.append(status)
        trajectories.append(target)
        records.append({"current_frame_token": token, "split": "navtest",
            "recording_group": frames[3].get("log_name", Path(images[-1]).parent.parent.name),
            "image_paths": images, "metric_cache_file": str(metric_loader.metric_cache_paths[token])})
    image_paths = sorted({path for record in records for path in record["image_paths"]})
    image_index = {path: index for index, path in enumerate(image_paths)}
    cache = np.lib.format.open_memmap(output_root / "rgb_frames.npy", mode="w+", dtype=np.uint8,
        shape=(len(image_paths), 128, 128, 3))
    def resize_image(index_and_path):
        index, path = index_and_path
        with Image.open(path) as image:
            cache[index] = cv2.resize(np.asarray(image.convert("RGB"))[28:-28], (128, 128), interpolation=cv2.INTER_AREA)
    with ThreadPoolExecutor(max_workers=8) as executor:
        for _ in executor.map(resize_image, enumerate(image_paths)):
            pass
    cache.flush()
    for record in records:
        record["frame_cache_indices"] = [image_index[path] for path in record.pop("image_paths")]
    np.savez(output_root / "planning_targets.npz", ego_status=np.stack(statuses), ego_trajectory_target=np.stack(trajectories))
    write_json(output_root / "manifest.json", {"records": records, "unique_observed_images": len(image_paths),
        "scope": "All 12146 official navtest scenes, evaluation only, four observed images",
        "targets_sha256": digest(output_root / "planning_targets.npz"), "cache_sha256": digest(output_root / "rgb_frames.npy")})
    print("NAVTEST_INPUTS_READY", len(records), flush=True)


def predict(arguments, specification):
    import torch
    from train_lpwm_full_planning import load_training_inputs, make_planning_inputs, PlanningFineTunedLPWM
    from planning_aware_future_prediction.object_centric.lpwm_candidate_planner import build_planning_model
    from run_lpwm_navsim_posttraining import check_gpu_reserve
    torch.set_num_threads(4)
    torch.cuda.set_device(arguments.gpu)
    device = torch.device(f"cuda:{arguments.gpu}")
    _, _, stage1_root, initial_checkpoint, manifest, targets, frames, _ = load_training_inputs(arguments.config, arguments.condition)
    output_root = PROJECT_ROOT / specification["output_directory"]
    condition_root = output_root / arguments.condition
    initial_evaluation = getattr(arguments, "initial", False)
    checkpoint_path = condition_root / "checkpoint.pt"
    if initial_evaluation:
        assert arguments.subset == "development"
        training = {"checkpoint_sha256": "initial-from-" + digest(initial_checkpoint)}
    else:
        training = json.loads((condition_root / "training_summary.json").read_text())
        assert not training["profile_only"] and training["epochs"] == specification["epochs"]
        assert digest(checkpoint_path) == training["checkpoint_sha256"]
    if arguments.subset == "navtest":
        cache_root = output_root / "navtest_inputs"
        manifest = json.loads((cache_root / "manifest.json").read_text())
        frames = np.load(cache_root / "rgb_frames.npy", mmap_mode="r")
        with np.load(cache_root / "planning_targets.npz") as stored:
            targets = {name: stored[name] for name in stored.files}
    records = manifest["records"]
    indices = np.array([index for index, record in enumerate(records) if record["split"] == arguments.subset])
    assert len(indices) == (12146 if arguments.subset == "navtest" else 27076)
    intervention_suffix = "" if arguments.intervention == "none" else "__" + arguments.intervention
    destination = condition_root / "evaluations" / (arguments.subset + intervention_suffix + ("__initial" if initial_evaluation else ""))
    destination.mkdir(parents=True, exist_ok=True)
    identity = {"checkpoint_sha256": training["checkpoint_sha256"], "subset": arguments.subset,
        "intervention": arguments.intervention, "expected_scenes": len(indices), "configuration_sha256": digest(arguments.config)}
    if (destination / "identity.json").exists():
        assert json.loads((destination / "identity.json").read_text()) == identity
    else:
        write_json(destination / "identity.json", identity)
    if (destination / "results.json").exists():
        return
    torch.manual_seed(specification["seed"])
    model = build_planning_model(initial_checkpoint, specification, arguments.condition, PROJECT_ROOT).to(device)
    if not initial_evaluation:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True), strict=True)
    teacher_metrics = None
    if specification.get("planner_architecture") == "particle_candidate_metrics" and arguments.subset == "development":
        teacher_metrics = np.load(PROJECT_ROOT / specification["teacher_directory"] / "candidate_metrics.npy", mmap_mode="r")
    model.eval()
    rows = []
    if (destination / "predictions.jsonl").exists():
        rows = [json.loads(line) for line in (destination / "predictions.jsonl").read_text().splitlines()]
    completed = {row["token"] for row in rows}
    assert len(completed) == len(rows)
    pending = [index for index in indices if records[index]["current_frame_token"] not in completed]
    started = time.monotonic()
    with (destination / "predictions.jsonl").open("a") as stream, torch.inference_mode():
        for offset in range(0, len(pending), 4):
            if offset % 128 == 0:
                check_gpu_reserve(arguments.gpu)
            chosen = np.array(pending[offset:offset + 4])
            observed, status, target = make_planning_inputs(records, chosen, frames, targets, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(observed, status, intervention=None if arguments.intervention == "none" else arguments.intervention)
            prediction = output["trajectory"].float()
            assert torch.isfinite(prediction).all()
            distances = (prediction[..., :2] - target[..., :2]).norm(dim=-1)
            for batch_index, index in enumerate(chosen):
                row = {"token": records[index]["current_frame_token"], "recording_group": records[index]["recording_group"],
                    "trajectory": prediction[batch_index].cpu().tolist(), "ade_meters": float(distances[batch_index].mean()),
                    "fde_meters": float(distances[batch_index, -1])}
                if "candidate_indices" in output:
                    row["candidate_index"] = int(output["candidate_indices"][batch_index])
                    if teacher_metrics is not None and np.isfinite(teacher_metrics[index]).all():
                        labels = torch.from_numpy(np.array(teacher_metrics[index, :, :6])).to(device)
                        row["candidate_metric_bce"] = float(torch.nn.functional.binary_cross_entropy_with_logits(output["metric_logits"][batch_index].float(), labels))
                    if "selected_refined_metric_logits" in output:
                        row["selected_refined_metric_logits"] = output["selected_refined_metric_logits"][batch_index].float().cpu().tolist()
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            if offset % 512 == 0:
                write_json(destination / "progress.json", {"completed": len(rows), "total": len(indices), "seconds": time.monotonic() - started})
    assert {row["token"] for row in rows} == {records[index]["current_frame_token"] for index in indices}
    write_json(destination / "results.json", {**identity, "windows": rows,
        "summary": {name: float(np.mean([row[name] for row in rows])) for name in ("ade_meters", "fde_meters")},
        "seconds_this_invocation": time.monotonic() - started})
    print("LPWM_FULL_PREDICTIONS_DONE", arguments.condition, arguments.subset, arguments.intervention, len(rows), flush=True)


def score(arguments, specification):
    import lzma
    import pickle
    from dataclasses import asdict
    from audit_official_drive_jepa_evaluation import official_configuration
    _, _, configuration, official_root = official_configuration(PROJECT_ROOT)
    from hydra.utils import instantiate
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    simulator, scorer = instantiate(configuration.simulator), instantiate(configuration.scorer)
    root = PROJECT_ROOT / specification["output_directory"]
    intervention_suffix = "" if arguments.intervention == "none" else "__" + arguments.intervention
    evaluation_name = arguments.subset + intervention_suffix + ("__initial" if getattr(arguments, "initial", False) else "")
    predictions_path = root / arguments.condition / "evaluations" / evaluation_name / "results.json"
    predictions = json.loads(predictions_path.read_text())
    by_token = {row["token"]: row for row in predictions["windows"]}
    if arguments.subset == "navtest":
        entries = [{**record, "token": record["current_frame_token"]} for record in
            json.loads((root / "navtest_inputs/manifest.json").read_text())["records"]]
        assert len(entries) == 12146 and set(by_token) == {entry["token"] for entry in entries}
    else:
        cache_manifest = (PROJECT_ROOT / specification["teacher_directory"] / "metric_cache_manifest.json") if "teacher_directory" in specification else (PROJECT_ROOT / "outputs/drive_jepa_selective_future/region_research_pdm_v1_20261003/metric_cache_manifest.json")
        all_entries = json.loads(cache_manifest.read_text())
        entries = [entry for entry in all_entries if entry["token"] in by_token]
        assert len(entries) > 0
        for entry in entries:
            if "cache_sha256" in entry:
                assert digest(entry["metric_cache_file"]) == entry["cache_sha256"]
    destination = PROJECT_ROOT / specification["shared_results_directory"] / "pdm" / f"{arguments.condition}__{evaluation_name}.json"
    prediction_digest = digest(predictions_path)
    if destination.exists():
        assert json.loads(destination.read_text())["prediction_sha256"] == prediction_digest
        return
    partial_path = root / arguments.condition / "evaluations" / evaluation_name / "pdm_scores.jsonl"
    rows = [json.loads(line) for line in partial_path.read_text().splitlines()] if partial_path.exists() else []
    completed = {row["token"] for row in rows}
    if specification.get("planner_architecture") == "particle_candidate_metrics":
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        from lpwm_refinement_oracle import initialize_oracle, score_refined_scene
        from prepare_lpwm_candidate_teacher import METRIC_NAMES
        pending_entries = [entry for entry in entries if entry["token"] not in completed]
        oracle_manifest = partial_path.parent / "evaluation_oracle_manifest.json"
        write_json(oracle_manifest, [{**entry, "index": index} for index, entry in enumerate(pending_entries)])
        jobs = [(index, np.asarray([by_token[entry["token"]]["trajectory"]], dtype=np.float32)) for index, entry in enumerate(pending_entries)]
        with ProcessPoolExecutor(max_workers=specification["teacher_workers"], mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize_oracle, initargs=(str(oracle_manifest),)) as executor, partial_path.open("a") as stream:
            for entry, (metric_values, _temporal) in zip(pending_entries, executor.map(score_refined_scene, jobs)):
                assert np.isfinite(metric_values).all()
                predicted = by_token[entry["token"]]
                row = {"token": entry["token"], "recording_group": predicted["recording_group"], **dict(zip(METRIC_NAMES, metric_values[0].tolist()))}
                if "selected_refined_metric_logits" in predicted:
                    logits = np.asarray(predicted["selected_refined_metric_logits"])
                    row["selected_refined_metric_bce"] = float(np.mean(np.logaddexp(0., logits) - metric_values[0, :6] * logits))
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
        completed = {row["token"] for row in rows}
    with partial_path.open("a") as stream:
        for entry in entries:
            if entry["token"] in completed:
                continue
            with lzma.open(entry["metric_cache_file"], "rb") as cache_stream:
                metric_cache = pickle.load(cache_stream)
            predicted = by_token[entry["token"]]
            trajectory = Trajectory(np.asarray(predicted["trajectory"], dtype=np.float32))
            metrics = asdict(pdm_score(metric_cache, trajectory, simulator.proposal_sampling, simulator, scorer))
            assert all(np.isfinite(value) for value in metrics.values())
            row = {"token": entry["token"], "recording_group": predicted["recording_group"], **metrics}
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
    assert len(rows) == len(entries) and len({row["token"] for row in rows}) == len(rows)
    summary = {name: float(np.mean([row[name] for row in rows])) for name in rows[0] if name not in ("token", "recording_group")}
    write_json(destination, {"windows": rows, "summary": summary, "scenes": len(rows), "prediction_sha256": prediction_digest,
        "scope": "official full navtest" if arguments.subset == "navtest" else "all eligible cached strict-navtrain development tokens",
        "scorer_source": str(official_root)})
    print("LPWM_FULL_PDM_DONE", arguments.condition, arguments.subset, summary, flush=True)


def summarize(specification):
    from summarize_lpwm_posttraining import paired_recording_interval
    root = PROJECT_ROOT / specification["output_directory"]
    result_root = PROJECT_ROOT / specification["shared_results_directory"]
    reports = {}
    for condition in specification["conditions"]:
        reports[condition] = {}
        for subset in ("development", "navtest"):
            inference = json.loads((root / condition / "evaluations" / subset / "results.json").read_text())
            pdm = json.loads((result_root / "pdm" / f"{condition}__{subset}.json").read_text())
            reports[condition][subset] = {**inference["summary"], "pdms_percent": 100 * pdm["summary"]["score"], "pdm_scenes": pdm["scenes"]}
        control = json.loads((root / condition / "evaluations/development__persistent_future/results.json").read_text())
        actual = json.loads((root / condition / "evaluations/development/results.json").read_text())
        def paired_rows(rows):
            return [{"token": row["token"], "recording_group": row["recording_group"], "metrics": {"ade": row["ade_meters"]}} for row in rows]
        reports[condition]["forecast_vs_persistent_future_ade"] = paired_recording_interval(paired_rows(actual["windows"]), paired_rows(control["windows"]), "ade")
    baseline = json.loads((PROJECT_ROOT / "results/official_drive_jepa_reproduction/full_navtest_results.json").read_text())
    paired_comparisons = {}
    if specification.get("planner_architecture") == "particle_candidate_metrics":
        for subset in ("development", "navtest"):
            def score_rows(condition):
                rows = json.loads((result_root / "pdm" / f"{condition}__{subset}.json").read_text())["windows"]
                return [{"token": row["token"], "recording_group": row["recording_group"], "metrics": {"pdms": row["score"] * 100}} for row in rows]
            for condition, reference in (("metric_plus_world", "imitation_plus_world"), ("metric_refinement_plus_world", "metric_plus_world")):
                paired_comparisons[subset + ":" + condition + "-" + reference] = paired_recording_interval(score_rows(condition), score_rows(reference), "pdms")
        for condition in specification["conditions"]:
            gate = json.loads((root / condition / "validation_gate.json").read_text())
            reports[condition]["development_validation"] = gate
    report = {"conditions": reports, "drive_jepa_unchanged_reference": baseline["all_server_percent_metrics"],
        "paired_pdms_percentage_point_comparisons": paired_comparisons,
        "scope": "One seed per registered condition; development was used for adaptation diagnostics. All navtest scenes evaluated after fixed final training epochs.",
        "performance_improvement_is_not_assumed": True}
    write_json(result_root / "summary.json", report)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    names = list(reports)
    axes[0].bar(names, [reports[name]["navtest"]["pdms_percent"] for name in names])
    axes[0].set(title="Official full navtest", ylabel="PDMS (%)", ylim=(0, 100))
    axes[1].bar(names, [reports[name]["navtest"]["ade_meters"] for name in names])
    axes[1].set(title="Full navtest trajectory error", ylabel="ADE (m)")
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=12)
    figure.tight_layout()
    figure.savefig(result_root / "planning_summary.png", dpi=160)
    plt.close(figure)
    print("LPWM_PLANNING_REPORT_DONE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/lpwm_planning/full_joint_training_v1.json")
    parser.add_argument("--mode", choices=("prepare-navtest", "predict", "score", "summarize"), required=True)
    parser.add_argument("--condition")
    parser.add_argument("--initial", action="store_true")
    parser.add_argument("--subset", choices=("development", "navtest"), default="development")
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=0)
    parser.add_argument("--intervention", choices=("none", "persistent_future"), default="none")
    arguments = parser.parse_args()
    arguments.config = arguments.config.resolve()
    specification = json.loads(arguments.config.read_text())
    if arguments.mode == "prepare-navtest":
        prepare_navtest(specification)
    elif arguments.mode == "predict":
        predict(arguments, specification)
    elif arguments.mode == "score":
        score(arguments, specification)
    else:
        summarize(specification)
