"""Show unmodified source RGB alongside the actual cached LPWM camera inputs."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = (("CAM_F0", "전방"), ("CAM_B0", "후방"), ("CAM_L0", "좌측"), ("CAM_R0", "우측"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / "manifest.json").exists(), "Preserve completed visualization"
    panel_path = ROOT / "outputs/lpwm_drivor_planning_path_representation_monitor_v1/panel.json"
    panel = json.loads(panel_path.read_text())
    record = next(record for record in panel["records"] if record["scene_type"] == "straight")
    cache_path = Path(record["cache_directory"]) / "images.npy"
    cached_images = np.load(cache_path, mmap_mode="r")[record["cache_row"]]
    assert cached_images.shape == (4, 128, 128, 3) and cached_images.dtype == np.uint8
    font_path = subprocess.check_output(["fc-match", "-f", "%{file}", "Noto Sans CJK KR"], text=True).strip()
    fonts = {size: ImageFont.truetype(font_path, size) for size in (16, 18, 21, 26, 32)}
    canvas = Image.new("RGB", (1288, 1952), "#f3f5f8")
    drawing = ImageDraw.Draw(canvas)
    drawing.text((24, 12), "LPWM 실제 카메라 입력: 원본 → 128×128", font=fonts[32], fill="#15243c")
    drawing.text((24, 58), "동일한 NAVSIM 학습 장면 · 4개 카메라의 현재 프레임 · 캐시에서 읽은 실제 입력", font=fonts[21], fill="#45556d")
    drawing.text((24, 103), "원본 영상  |  원래 비율 유지하여 축소 표시", font=fonts[21], fill="#15243c")
    drawing.text((694, 103), "128×128 입력", font=fonts[21], fill="#15243c")
    drawing.text((880, 103), "입력 3배 확대  |  최근접 보간", font=fonts[21], fill="#15243c")
    camera_records = []
    gallery_sections = []
    for camera_index, (camera_name, camera_label) in enumerate(CAMERAS):
        source = Path(record["current_camera_paths"][camera_index])
        with Image.open(source) as image:
            original = image.convert("RGB")
        expected = np.asarray(original.resize((128, 128), Image.Resampling.BICUBIC))
        assert np.array_equal(expected, cached_images[camera_index]), f"Cache mismatch: {camera_name}"
        original_name = f"original_{camera_name}{source.suffix}"
        input_name = f"input_{camera_name}_128x128.png"
        shutil.copy2(source, output / original_name)
        assert digest(source) == digest(output / original_name)
        processed = Image.fromarray(np.array(cached_images[camera_index]))
        processed.save(output / input_name)
        top = 144 + camera_index * 438
        drawing.rounded_rectangle((12, top, 1276, top+426), radius=10, fill="white")
        drawing.text((24, top+6), f"{camera_label}  {camera_name}  ·  원본 {original.width}×{original.height}", font=fonts[26], fill="#15243c")
        displayed_original = original.copy()
        displayed_original.thumbnail((640, 384), Image.Resampling.LANCZOS)
        canvas.paste(displayed_original, (24 + (640-displayed_original.width)//2, top+40+(384-displayed_original.height)//2))
        canvas.paste(processed, (704, top+40+128))
        canvas.paste(processed.resize((384, 384), Image.Resampling.NEAREST), (880, top+40))
        drawing.text((690, top+304), "실제 저장 픽셀", font=fonts[18], fill="#45556d")
        drawing.text((690, top+332), "전체 영상 축소", font=fonts[18], fill="#45556d")
        drawing.text((690, top+360), "crop 없음", font=fonts[18], fill="#45556d")
        camera_records.append({"camera": camera_name, "source": str(source), "source_sha256": digest(source),
            "original_size_width_height": list(original.size), "original_copy": original_name,
            "preprocessed_image": input_name, "cached_pixels_match_recomputed_preprocessing": True,
            "cached_rgb_sha256": hashlib.sha256(cached_images[camera_index].tobytes()).hexdigest()})
        gallery_sections.append(f'<h2>{camera_label} {camera_name}</h2><p><a href="{original_name}">원본 해상도로 열기</a> · '
            f'<a href="{input_name}">실제128×128 이미지</a></p><div class="pair"><a href="{original_name}"><img src="{original_name}" alt="원본"></a>'
            f'<a href="{input_name}"><img class="input" src="{input_name}" alt="전처리된입력"></a></div>')
    drawing.text((24, 1903), f"Scene token: {record['token']}  |  모델에는 RGB / 255, float32로 전달됩니다.", font=fonts[18], fill="#45556d")
    drawing.text((24, 1929), "오른쪽 확대는 보기 위한 표시이며, 모델 입력 해상도는 128×128입니다.", font=fonts[16], fill="#45556d")
    canvas.save(output / "four_camera_original_vs_input.png")
    manifest = {"scene": record, "selection": "First straight-trajectory scene in the existing fixed96 panel, selected without inspecting images",
        "panel_sha256": digest(panel_path), "camera_order": [name for name, _ in CAMERAS],
        "cache_path": str(cache_path), "cache_row": record["cache_row"], "cameras": camera_records,
        "resize": "PIL BICUBIC full RGB image to128x128; no crop", "model_tensor_shape_without_batch": [4, 3, 128, 128],
        "model_pixel_conversion": "uint8 RGB -> float32 RGB/255; native LPWM normalize_rgb=False",
        "overview_original_display": "fit within640x384 preserving aspect ratio; full-resolution originals copied byte-for-byte",
        "overview_enlarged_input": "3x nearest-neighbor display only", "uses_gpu": False, "changes_training": False,
        "source_sha256": digest(__file__)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "index.html").write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><title>LPWM 실제 카메라 입력</title>
<style>body{font-family:sans-serif;max-width:1400px;margin:24px auto;padding:0 16px;color:#15243c}img{max-width:100%;height:auto}.pair{display:grid;grid-template-columns:2fr 1fr;gap:24px;align-items:center}.input{width:384px;image-rendering:pixelated}</style>
<h1>원본과 실제128×128 LPWM 입력</h1><p>이미지를클릭하면각파일의실제해상도로열립니다. 원본은바이트그대로복사했고입력은학습캐시에서읽었습니다.
모든4카메라에서원본의BICUBIC128×128변환과캐시픽셀이완전히같음을확인했습니다.</p>
<p>Scene token: ''' + html.escape(record["token"]) + '''</p><p><a href="four_camera_original_vs_input.png">전체비교그림</a> · <a href="manifest.json">장면·전처리근거</a></p>'''
        + "".join(gallery_sections) + "</html>")
    print(json.dumps({"output": str(output), "scene_token": record["token"], "all_four_cached_images_verified": True,
        "original_sizes": [camera["original_size_width_height"] for camera in camera_records]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/lpwm_camera_input_visualization_v1")
    run(parser.parse_args().output.resolve())
