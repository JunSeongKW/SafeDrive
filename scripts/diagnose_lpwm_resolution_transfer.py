"""CPU-only paired resolution transfer; no optimization or active-run changes.

Uses one NAVSIM-trained 128-square checkpoint and 64 encoded/30 decoded
foreground particles throughout. Decoder-only transfer holds every encoded
latent and particle selection fixed. This is an immediate transfer test, not
a matched retraining experiment or a planning benchmark.
"""
from contextlib import redirect_stdout
from datetime import datetime
import io
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as functional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from planning_aware_future_prediction.object_centric.lpwm_bridge import load_official_lpwm, checkpoint_digest
from planning_aware_future_prediction.object_centric.lpwm_rectangular import load_rectangular_lpwm
from train_lpwm_reduced_particle_stage1 import load_clip, prepare_records
from visualize_small_corpus_lpwm_rgb import render_comparison

CHECKPOINT = ROOT / "outputs/lpwm_navsim_full_posttraining_v2/stage1/checkpoint.pt"
RUN = ROOT / "outputs/four_model_small_corpus_v1/lpwm_ssl"
DESTINATION = ROOT / "results/four_model_small_corpus_v1/resolution_transfer_diagnosis"
ARRAY_DESTINATION = ROOT / "outputs/four_model_small_corpus_v1/diagnostics/resolution_transfer"


def decode_observed_frame(model, encoded):
    attributes = [encoded[key][:, 1:2].contiguous() for key in
                  ("z", "z_scale", "z_features", "obj_on", "z_depth", "z_bg_features")]
    filter_key = encoded["z_base_var"][:, 1:2].sum(-1).contiguous()
    decoded = model.decode_all(*attributes, z_ctx=encoded["z_context"], filter_key=filter_key)
    return {key: decoded[key].reshape(-1, 3, *decoded[key].shape[-2:])[0].clone()
            for key in ("rec_rgb", "bg_rgb")}


def display_image(tensor):
    if tensor.shape[-2:] == (128, 128):
        tensor = functional.interpolate(tensor[None], size=(256, 512), mode="nearest")[0]
    return tensor.numpy().copy()


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    ARRAY_DESTINATION.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    torch.manual_seed(47)
    started = time.monotonic()
    with redirect_stdout(io.StringIO()):
        native, _ = load_official_lpwm("cpu", CHECKPOINT)
        rectangular, _ = load_rectangular_lpwm("cpu", foreground_particles=64)
    # Keep spatial buffers made for the new canvas, but copy every trained weight.
    spatial_buffers = {name: value.clone() for name, value in rectangular.named_buffers()
                       if name.rsplit(".", 1)[-1] in ("patch_centers", "particle_anchors", "particles_anchor")}
    rectangular.load_state_dict(native.state_dict(), strict=True)
    with torch.no_grad():
        for name, value in rectangular.named_buffers():
            if name in spatial_buffers:
                value.copy_(spatial_buffers[name])
    for module in (rectangular, rectangular.encoder_module, rectangular.encoder_module.particle_enc):
        module.n_kp_dec = 30
    rectangular.decoder_module.n_kp_enc = 30
    for model in (native, rectangular):
        model.eval().requires_grad_(False)
        model.timestep_horizon = 7
        assert model.n_kp_enc == 64 and model.n_kp_dec == 30
        assert not model.normalize_rgb
    native_parameters = dict(native.named_parameters())
    rectangular_parameters = dict(rectangular.named_parameters())
    assert native_parameters.keys() == rectangular_parameters.keys()
    assert all(torch.equal(value, rectangular_parameters[name]) for name, value in native_parameters.items())
    registration = json.loads((RUN / "registration.json").read_text())
    _, validation = prepare_records(Path(registration["manifest"]), 32)
    historical = ROOT / "results/four_model_small_corpus_v1/historical_rgb_same_scenes"
    historical_report = json.loads((historical / "comparison_report.json").read_text())
    prior_outputs = np.load(historical / "historical_native_outputs.npz")
    scene_rows, metric_rows, arrays = [], [], {}
    with torch.inference_mode():
        for scene_index, record in enumerate(validation[:3]):
            rectangular_clip = load_clip(record["paths"])[None].float().div(255)
            square_images = []
            for path in record["paths"]:
                with Image.open(path) as opened:
                    square_images.append(cv2.resize(np.asarray(opened.convert("RGB"))[28:-28],
                                                   (128, 128), interpolation=cv2.INTER_AREA))
            square_clip = torch.from_numpy(np.stack(square_images)).permute(0, 3, 1, 2).contiguous().float().div(255)[None]
            native_encoded = native.encode_all(square_clip, deterministic=True)
            native_decoded = decode_observed_frame(native, native_encoded)
            original_output = torch.from_numpy(prior_outputs[f"scene{scene_index}_stage1_reconstruction_native"])
            reproduction_error = float((native_decoded["rec_rgb"] - original_output).abs().max())
            assert reproduction_error < 1e-6, reproduction_error
            decoder_transfer = decode_observed_frame(rectangular, native_encoded)
            rectangular_encoded = rectangular.encode_all(rectangular_clip, deterministic=True)
            full_transfer = decode_observed_frame(rectangular, rectangular_encoded)
            encoder_transfer = decode_observed_frame(native, rectangular_encoded)
            conditions = dict(native128=native_decoded, decoder_only512=decoder_transfer,
                              encoder_only512=encoder_transfer, full512=full_transfer)
            scene_metrics = {}
            for condition, decoded in conditions.items():
                prediction = decoded["rec_rgb"]
                common_prediction = functional.interpolate(prediction[None], size=(128, 128), mode="area")[0]
                scene_metrics[condition] = float((common_prediction - square_clip[0, 1]).square().mean())
                for output_key, output in decoded.items():
                    arrays[f"scene{scene_index}_{condition}_{output_key}"] = output.numpy().copy()
            input_resize_mse = float((functional.interpolate(rectangular_clip[:, 1], size=(128, 128), mode="area")[0]
                                     - square_clip[0, 1]).square().mean())
            seen = historical_report["scenes"][scene_index]["historical_stage1_training_image_overlap"]
            exposure = "Seen in old training" if seen else "Held out from old training"
            row = dict(caption=f"Scene {scene_index + 1} | {record['recording']} | {exposure}",
                       input=rectangular_clip[0, 1].numpy().copy())
            row.update({condition: display_image(decoded["rec_rgb"]) for condition, decoded in conditions.items()})
            scene_rows.append(row)
            metric_rows.append(dict(scene=scene_index + 1, recording=record["recording"], raw_frame_paths=record["paths"],
                                   old_training_image=seen, mse_common128=scene_metrics,
                                   native_reproduction_max_abs=reproduction_error, input_resize_mse=input_resize_mse))
            print(json.dumps(dict(scene=scene_index + 1, mse_common128=scene_metrics,
                                  elapsed_seconds=time.monotonic() - started)), flush=True)
    prior_outputs.close()
    averages = {condition: float(np.mean([row["mse_common128"][condition] for row in metric_rows]))
                for condition in conditions}
    ratios = {condition: value / averages["native128"] for condition, value in averages.items()}
    report = dict(created_at=datetime.now().astimezone().isoformat(), checkpoint=str(CHECKPOINT),
                  checkpoint_sha256=checkpoint_digest(CHECKPOINT), device="cpu", threads=2,
                  encoded_foreground_particles=64, decoded_foreground_particles=30,
                  parameter_tensor_count=len(native_parameters), all_learned_parameters_exactly_identical=True,
                  spatial_buffers_regenerated=list(spatial_buffers), scenes=metric_rows,
                  mean_mse_common128=averages, mse_ratio_vs_native128=ratios,
                  condition_definition=dict(native128="Original 128 encoder and decoder",
                      decoder_only512="Identical native128 latent tensors and particle selection; rectangular512 decoder",
                      encoder_only512="Rectangular512 encoder; original128 decoder",
                      full512="Rectangular512 encoder and decoder, still 64 encoded/30 decoded"),
                  metric="Outputs area-resized to the same 128x128 target before MSE; no clamping before metrics",
                  limitations=["3 fixed examples, not a held-out dataset benchmark; first 2 appeared in old training",
                      "Immediate resolution transfer without retraining; cannot assign cause of the current retrained16 model",
                      "Encoder transfer includes pooling, prior patches, masks, anchors and input interpolation changes",
                      "Common128 metric omits the possible benefit of extra detail at native512 resolution",
                      "Observed reconstruction only; not causal future prediction or planning evaluation"],
                  training_updates=0, active_training_changed=False)
    array_path = ARRAY_DESTINATION / "outputs_native_resolution.npz"
    np.savez_compressed(array_path, **arrays)
    report["native_output_arrays"] = str(array_path)
    report["native_output_arrays_sha256"] = checkpoint_digest(array_path)
    (DESTINATION / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    footer = ("Same trained weights and 64 encoded / 30 decoded foreground particles. No additional training.\n"
              "128 outputs enlarged by nearest pixels for display. This diagnoses immediate transfer, not final adapted performance.")
    render_comparison(DESTINATION / "decoder_only_resolution_transfer.png",
                      "LPWM | Isolating the resolution extension of the decoder",
                      "Middle and right receive EXACTLY the same particle positions, appearance and background latent.",
                      [("Observed image", "input"), ("Original decoder: 128x128", "native128"),
                       ("Extended decoder: 512x256", "decoder_only512")], scene_rows, footer)
    render_comparison(DESTINATION / "encoder_and_decoder_resolution_transfer.png",
                      "LPWM | Immediate resolution transfer with identical learned weights",
                      "Reconstruction only. Original128, encoder-only extension, and full512 extension are shown separately.",
                      [("Observed image", "input"), ("Original encoder + decoder", "native128"),
                       ("512 encoder + 128 decoder", "encoder_only512"), ("512 encoder + 512 decoder", "full512")], scene_rows, footer)
    print(json.dumps(dict(complete=True,output=str(DESTINATION),mean_mse_common128=averages,
                          mse_ratio_vs_native128=ratios)), flush=True)


if __name__ == "__main__":
    main()
