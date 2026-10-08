"""Render fixed planning particle snapshots and diagnose saved scores on CPU.

This reads saved arrays only. It does not run a model or change training.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plot
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment


BEFORE_COLOR = (0, 192, 220)
AFTER_COLOR = (245, 158, 11)
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
COMMAND_NAMES = {0: "좌회전", 1: "직진", 2: "우회전"}


def font(size):
    return ImageFont.truetype(FONT_PATH, size)


def geometry(attributes):
    assert attributes.shape == (16, 28) and np.isfinite(attributes).all()
    centers = (attributes[:, :2][:, ::-1] + 1) * [256, 128] - .5
    sizes = attributes[:, 2:4][:, ::-1] * [512, 256]
    presence = attributes[:, 4]
    assert ((presence >= 0) & (presence <= 1)).all()
    assert (sizes > 0).all()
    return centers, sizes, presence


def change_statistics(initial_attributes, later_attributes):
    initial_centers, initial_sizes, initial_presence = geometry(initial_attributes)
    later_centers, later_sizes, later_presence = geometry(later_attributes)
    same_index_shift = np.linalg.norm(later_centers - initial_centers, axis=1)
    cost = np.linalg.norm(initial_centers[:, None] - later_centers[None], axis=-1)
    initial_indices, later_indices = linear_sum_assignment(cost)
    relative_size_change = np.abs(later_sizes - initial_sizes) / initial_sizes
    return dict(
        mean_same_index_center_shift_px=float(same_index_shift.mean()),
        maximum_same_index_center_shift_px=float(same_index_shift.max()),
        mean_geometrically_matched_center_shift_px=float(cost[initial_indices, later_indices].mean()),
        mean_absolute_width_height_change_percent=float(relative_size_change.mean() * 100),
        mean_presence_before=float(initial_presence.mean()), mean_presence_after=float(later_presence.mean()),
        presence_above_point95_before=int((initial_presence > .95).sum()),
        presence_above_point95_after=int((later_presence > .95).sum()),
        initial_indices=initial_indices.tolist(), later_indices=later_indices.tolist(),
        matching_note="Same array index is not verified object identity. Hungarian matching uses only center distance.",
    )


def draw_particles(panel, attributes, color, opacity=205):
    centers, sizes, presence_values = geometry(attributes)
    layer = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    drawing = ImageDraw.Draw(layer)
    for center, size, presence in zip(centers, sizes, presence_values):
        horizontal, vertical = map(float, center)
        width, height = map(float, size)
        alpha = int(45 + (opacity - 45) * presence)
        drawing.rectangle((horizontal-width/2, vertical-height/2,
                           horizontal+width/2, vertical+height/2), outline=(*color, alpha), width=1)
        radius = float(np.sqrt(4 + 25 * presence))
        drawing.ellipse((horizontal-radius-.5, vertical-radius-.5,
                         horizontal+radius+.5, vertical+radius+.5), fill=(255, 255, 255, alpha))
        drawing.ellipse((horizontal-radius, vertical-radius,
                         horizontal+radius, vertical+radius), fill=(*color, alpha))
    return Image.alpha_composite(panel.convert("RGBA"), layer).convert("RGB")


def draw_overlay(pixels, before, after):
    panel = draw_particles(Image.fromarray(pixels), before, BEFORE_COLOR, 155)
    initial_centers, _, _ = geometry(before)
    later_centers, _, _ = geometry(after)
    cost = np.linalg.norm(initial_centers[:, None] - later_centers[None], axis=-1)
    initial_indices, later_indices = linear_sum_assignment(cost)
    drawing = ImageDraw.Draw(panel)
    for initial_index, later_index in zip(initial_indices, later_indices):
        start = initial_centers[initial_index]
        end = later_centers[later_index]
        displacement = end - start
        length = np.linalg.norm(displacement)
        if length < 4:
            continue
        direction = displacement / length
        normal = np.array([-direction[1], direction[0]])
        drawing.line([tuple(start), tuple(end)], fill="white", width=3)
        drawing.line([tuple(start), tuple(end)], fill="#bc37b3", width=1)
        drawing.polygon([tuple(end), tuple(end-direction*5+normal*2.5),
                         tuple(end-direction*5-normal*2.5)], fill="#bc37b3")
    return draw_particles(panel, after, AFTER_COLOR, 205)


def compose(destination, title, subtitle, headings, rows, footer):
    margin, gap, panel_width, panel_height = 20, 14, 512, 256
    header_height, row_height = 125, 310
    width = margin*2 + len(headings)*panel_width + (len(headings)-1)*gap
    height = header_height + len(rows)*row_height + 100
    canvas = Image.new("RGB", (width, height), "#f3f5f7")
    drawing = ImageDraw.Draw(canvas)
    drawing.text((margin, 10), title, font=font(28), fill="#17212e")
    drawing.text((margin, 49), subtitle, font=font(18), fill="#344254")
    for column, heading in enumerate(headings):
        drawing.text((margin+column*(panel_width+gap), 86), heading, font=font(21), fill="#17212e")
    for row_index, (caption, panels) in enumerate(rows):
        assert len(panels) == len(headings)
        top = header_height + row_index*row_height
        drawing.text((margin, top), caption, font=font(17), fill="#344254")
        for column, panel in enumerate(panels):
            assert panel.size == (512, 256)
            canvas.paste(panel, (margin+column*(panel_width+gap), top+30))
    drawing.multiline_text((margin, height-93), footer, font=font(17), fill="#344254", spacing=5)
    canvas.save(destination)


def main(arguments):
    study = arguments.study_directory.resolve()
    destination = arguments.output_directory.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    source_paths = []
    corpus_path = study/'corpus'/'manifest.json'
    corpus = json.loads(corpus_path.read_text())
    development_tokens = [record['token'] for record in corpus['records'] if record['study_split'] == 'dev']
    assert len(development_tokens) == 1024
    source_paths.append(corpus_path)
    planning_bundles, changes, evaluations, scene_commands = {}, {}, {}, {}
    visualized_scene_tokens = {}
    conditions = {"lpwm_sequential": ("LPWM 2-stage", [0, 1, 3, 5]),
                  "lpwm_joint": ("LPWM 공동학습", [0, 1, 3])}
    for kind, (display_name, epochs) in conditions.items():
        planning_bundles[kind], changes[kind], evaluations[kind] = {}, {}, {}
        for epoch in epochs:
            label = "before" if epoch == 0 else f"pass{epoch}"
            path = study/kind/"particles"/f"{label}.npz"
            source_paths.append(path)
            with np.load(path, allow_pickle=False) as archive:
                planning_bundles[kind][epoch] = {name: archive[name].copy() for name in archive.files}
            for scene_index in range(3):
                image_key = f"scene{scene_index}_image"
                assert np.array_equal(planning_bundles[kind][0][image_key], planning_bundles[kind][epoch][image_key])
            if epoch:
                score_path = study/kind/"validation"/f"pass{epoch}.pdms.json"
                source_paths.append(score_path)
                evaluations[kind][epoch] = json.loads(score_path.read_text())
            scene_changes = [change_statistics(planning_bundles[kind][0][f"scene{scene}_attributes"],
                                              planning_bundles[kind][epoch][f"scene{scene}_attributes"])
                             for scene in range(3)]
            changes[kind][epoch] = dict(per_scene=scene_changes,
                mean_same_index_center_shift_px=float(np.mean([row["mean_same_index_center_shift_px"] for row in scene_changes])),
                mean_geometrically_matched_center_shift_px=float(np.mean([row["mean_geometrically_matched_center_shift_px"] for row in scene_changes])),
                mean_absolute_width_height_change_percent=float(np.mean([row["mean_absolute_width_height_change_percent"] for row in scene_changes])),
                mean_presence=float(np.mean([row["mean_presence_after"] for row in scene_changes])),
                presence_above_point95_count=sum(row["presence_above_point95_after"] for row in scene_changes))
        latest_epoch = epochs[-1]
        scores_path = study/kind/"validation"/f"pass{latest_epoch}.scores.csv"
        with scores_path.open() as stream:
            scene_rows = list(csv.DictReader(stream))[:3]
        prediction_path = study/kind/'validation'/f'pass{latest_epoch}.npz'
        with np.load(prediction_path, allow_pickle=False) as predictions:
            assert predictions['tokens'].tolist() == development_tokens
        assert [row['token'] for row in scene_rows] == development_tokens[:3]
        visualized_scene_tokens[kind] = [row['token'] for row in scene_rows]
        source_paths += [scores_path, prediction_path]
        scene_commands[kind] = [COMMAND_NAMES[int(row["command"])] for row in scene_rows]
        rows = []
        for scene in range(3):
            pixels = planning_bundles[kind][0][f"scene{scene}_image"]
            panels = [Image.fromarray(pixels)] + [draw_particles(Image.fromarray(pixels),
                planning_bundles[kind][epoch][f"scene{scene}_attributes"], BEFORE_COLOR if epoch == 0 else AFTER_COLOR)
                for epoch in epochs]
            caption = f"고정 개발 장면 {scene+1} | 실제 명령: {scene_commands[kind][scene]} | 최신 PDMS: {float(scene_rows[scene]['score'])*100:.2f}"
            rows.append((caption, panels))
        initial_description = "SSL 5 epoch 이후, planning 학습 직전" if kind == "lpwm_sequential" else "공개 LPWM 초기화, 로컬 SSL/planning 학습 직전"
        compose(destination/f"{kind}_particle_timeline.png", f"{display_name}: 동일 장면의 particle 학습 변화",
            f"초기 = {initial_description}. 각 패널은 현재 프레임, 네이티브 512×256 좌표를 유지합니다.",
            ["입력 이미지"]+["planning 전" if epoch == 0 else f"planning {epoch} epoch" for epoch in epochs], rows,
            "점: 16개 foreground particle 중심. 점 크기: presence. 박스: 이미지 glimpse 범위이며 객체 검출 박스가 아닙니다.\n"
            "presence는 planner attention/주행 중요도가 아닙니다. background particle은 이 박스 그림에 포함되지 않습니다.\n"
            "학습 전에 고정된 첫 3개 개발 장면입니다. 객체 정보 보존이나 planning 인과효과를 입증하는 그림은 아닙니다.")

    for scene in range(3):
        assert np.array_equal(planning_bundles['lpwm_sequential'][0][f'scene{scene}_image'],
                              planning_bundles['lpwm_joint'][0][f'scene{scene}_image'])
        rows = []
        for kind, (display_name, epochs) in conditions.items():
            latest_epoch = epochs[-1]
            before = planning_bundles[kind][0][f"scene{scene}_attributes"]
            after = planning_bundles[kind][latest_epoch][f"scene{scene}_attributes"]
            pixels = planning_bundles[kind][0][f"scene{scene}_image"]
            values = changes[kind][latest_epoch]['per_scene'][scene]
            panels = [Image.fromarray(pixels), draw_particles(Image.fromarray(pixels), before, BEFORE_COLOR),
                      draw_particles(Image.fromarray(pixels), after, AFTER_COLOR), draw_overlay(pixels, before, after)]
            caption = (f"{display_name} | {latest_epoch} epoch | 같은 인덱스 중심 이동 평균 {values['mean_same_index_center_shift_px']:.1f}px "
                       f"| 폭·높이 변화 {values['mean_absolute_width_height_change_percent']:.1f}%")
            rows.append((caption, panels))
        compose(destination/f"scene{scene+1}_before_after_overlay.png", f"고정 개발 장면 {scene+1}: 입력 · 학습 전 · 학습 후 · 겹침",
            "청록색 = 각 조건의 초기 particle, 주황색 = 최신 particle. 두 조건의 학습 전 초기화는 서로 다릅니다.",
            ['입력 이미지','학습 전','학습 후 (순차5 / 공동3)','전후 겹침'], rows,
            "마젠타 화살표: 중심 거리로 매칭한 위치 차이. 객체 ID 추적이나 attention 변화가 아닙니다.\n"
            "점 크기는 presence, 박스는 glimpse 범위입니다. 실제 객체의 위치·크기·종류 정답을 표시한 그림이 아닙니다.\n"
            "모든 패널을 같은 512×256 영역으로 고정했습니다. 확대·축소·이미지 선명화 없이 저장된 RGB를 사용했습니다.")

    figure, axes = plot.subplots(2, 2, figsize=(11, 7.5))
    for kind, (display_name, epochs) in conditions.items():
        label = '2-stage' if kind == 'lpwm_sequential' else 'Joint'
        values = [changes[kind][epoch] for epoch in epochs]
        axes[0,0].plot(epochs, [row['mean_same_index_center_shift_px'] for row in values], 'o-', label=label)
        axes[0,1].plot(epochs, [row['mean_absolute_width_height_change_percent'] for row in values], 'o-', label=label)
        axes[1,0].plot(epochs, [row['mean_presence'] for row in values], 'o-', label=label)
        available = [epoch for epoch in epochs if epoch]
        axes[1,1].plot(available, [evaluations[kind][epoch]['pdms'] for epoch in available], 'o-', label=label)
    for kind, label in [('drivor','DrivoR'),('jepa','Drive-JEPA backbone')]:
        report_paths = [study/kind/'validation'/f'pass{epoch}.pdms.json' for epoch in [1,3,5]]
        source_paths += report_paths
        scores = [json.loads(path.read_text())['pdms'] for path in report_paths]
        axes[1,1].plot([1,3,5], scores, 'o--', label=label)
    for axis, title in zip(axes.flat, ['Same-index center change (input pixels)', 'Absolute width/height change (%)',
                                     'Presence mean (48 particles / 3 scenes)', 'PDMS (same 1,024 development scenes)']):
        axis.set_title(title); axis.set_xlabel('Planning epoch'); axis.set_xticks([0,1,3,5]); axis.grid(alpha=.25); axis.legend(fontsize=8)
    axes[1,0].set_ylim(0,1.04)
    figure.suptitle('Geometry changes do not establish object information or planning benefit')
    figure.tight_layout(); figure.savefig(destination/'particle_change_and_pdms_curves.png', dpi=150); plot.close(figure)

    ssl = study/'lpwm_ssl'
    ssl_before = np.load(ssl/'before_training_visuals.npz', allow_pickle=False)
    ssl_after = np.load(ssl/'after_training_visuals.npz', allow_pickle=False)
    source_paths += [ssl/'before_training_visuals.npz', ssl/'after_training_visuals.npz']
    rgb_rows, ssl_particle_rows = [], []
    ssl_epochs = [(0,'before_training'),(1,'update655'),(3,'update1965'),(5,'after_training')]
    ssl_bundles = {epoch:np.load(ssl/f'{label}_visuals.npz', allow_pickle=False) for epoch,label in ssl_epochs}
    source_paths += [ssl/f'{label}_visuals.npz' for _,label in ssl_epochs] + [ssl/'after_training.json']
    ssl_report = json.loads((ssl/'after_training.json').read_text())
    for scene in range(3):
        def rgb(channels_first):
            return Image.fromarray(np.rint(channels_first.transpose(1,2,0).clip(0,1)*255).astype(np.uint8))
        for key in ['input','truth']:
            assert np.array_equal(ssl_before[f'scene{scene}_{key}'], ssl_after[f'scene{scene}_{key}'])
        caption=f"별도 SSL 검증 clip {scene+1} | {ssl_report['records'][scene]['recording']} (planning 그림의 장면과 다릅니다)"
        rgb_rows.append((caption, [rgb(ssl_after[f'scene{scene}_{key}']) for key in ['input','reconstruction','truth','forecast']]))
        panels = [rgb(ssl_after[f'scene{scene}_input'])]
        for epoch,label in ssl_epochs:
            bundle=ssl_bundles[epoch]
            assert np.array_equal(bundle[f'scene{scene}_input'], ssl_after[f'scene{scene}_input'])
            attributes = np.zeros((16,28), dtype=np.float32)
            attributes[:,:2] = bundle[f'scene{scene}_positions']
            attributes[:,2:4] = bundle[f'scene{scene}_scales']
            attributes[:,4] = bundle[f'scene{scene}_presence'].reshape(16)
            panels.append(draw_particles(rgb(ssl_after[f'scene{scene}_input']), attributes, BEFORE_COLOR if epoch==0 else AFTER_COLOR))
        ssl_particle_rows.append((caption,panels))
    compose(destination/'ssl_final_reconstruction_and_future.png', 'LPWM Stage1: SSL 5 epoch 이후의 복원과 미래 예측',
        '미래 예측은 현재·직전 두 프레임만 관측합니다. 실제 미래는 비교용이며 모델 입력이 아닙니다.',
        ['현재 입력 (t=0)','현재 복원 (SSL 5 epoch)','실제 미래 (+3초)','예측 미래 (+3초)'], rgb_rows,
        '원본 네이티브 512×256 출력입니다. 선명화·보정하지 않았습니다.\n'
        '이 RGB 결과는 Stage1 검증이며 planning 이후의 복원 결과가 아닙니다.\n'
        '낮은 영상 MSE만으로 작은 객체·차선·도로 경계의 정보 보존을 입증할 수 없습니다.')
    compose(destination/'ssl_particle_training_timeline.png', 'LPWM Stage1: SSL 학습 중 particle 변화',
        'SSL 학습 전·1·3·5 epoch, 학습 전에 고정된 같은 세 검증 clip. Planning 검증 장면과는 다른 자료입니다.',
        ['현재 입력','SSL 전','SSL 1 epoch','SSL 3 epoch','SSL 5 epoch'],ssl_particle_rows,
        '점: foreground particle 중심, 크기: presence. 박스: glimpse 범위. 검출 박스나 planner attention이 아닙니다.\n'
        '전체 영상 복원/미래 예측만으로 학습한 Stage1 자료입니다. 객체 GT 감독을 추가하지 않았습니다.\n'
        '512×256 좌표를 고정해 저장된 배열을 그렸습니다. 그림은 3개 clip의 보조 진단입니다.')

    score_diagnostics = {}
    for kind, epoch in [('drivor',5),('jepa',5),('lpwm_sequential',5),('lpwm_joint',3)]:
        score_path=study/kind/'validation'/f'pass{epoch}.scores.csv'
        source_paths.append(score_path)
        with score_path.open() as stream:
            scores=list(csv.DictReader(stream))
        assert len(scores) == 1024 and all(row['valid'] == 'True' for row in scores)
        assert [row['token'] for row in scores] == development_tokens
        score_diagnostics[kind]=dict(epoch=epoch,count=len(scores),
            zero_score_count=sum(float(row['score']) == 0 for row in scores),
            non_drivable_trajectory_count=sum(float(row['drivable_area_compliance']) == 0 for row in scores),
            at_fault_collision_count=sum(float(row['no_at_fault_collisions']) == 0 for row in scores),
            mean_progress=float(np.mean([float(row['ego_progress']) for row in scores])))
    manifest = dict(created_at=datetime.now().astimezone().isoformat(), planning_geometry_changes=changes,
        visualized_scene_tokens=visualized_scene_tokens, prediction_and_score_token_order_verified=True,
        score_diagnostics=score_diagnostics, ssl_final_metrics=ssl_report['mean'],
        source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(set(source_paths))},
        no_model_inference=True,no_gpu_work=True,training_changed=False,
        image_selection='First3 fixed saved development scenes; SSL images are first3 independent fixed SSL validation clips.',
        geometry_scope='3 scenes x16 foreground particles. Background has no foreground box in these arrays.',
        interpretation='Position/presence change is not attention or object information preservation. Geometry-only matching does not establish object identity.',
        files=[str(path) for path in sorted(destination.glob('*.png'))])
    (destination/'diagnosis.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(output_directory=str(destination),changes={kind:changes[kind][epochs[-1]] for kind,(_,epochs) in conditions.items()},
                         score_diagnostics=score_diagnostics),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--study-directory',type=Path,required=True)
    parser.add_argument('--output-directory',type=Path,required=True)
    main(parser.parse_args())
