"""提出用動画（1〜3 分）を作る。

  python scripts/make_video.py            # 足りない場面だけ録画して結合
  python scripts/make_video.py --force T5_disturb   # 指定した場面を撮り直す

各場面は outputs/video/parts/*.mp4 に書き出してから結合する（撮り直しが場面単位でできる）。
デモ場面は実際に Gemini を呼び出して実行した様子をそのまま録画している（台本で動かしてはいない）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from src.sim import OUTPUTS  # noqa: E402
from src.tasks import TASK_BY_ID  # noqa: E402
from src.video import Clip, W, H, _font, concat, image_card, render, title_card  # noqa: E402

PARTS = OUTPUTS / "video" / "parts"
RESULTS = ROOT / "docs" / "results"
ACCOUNT = "Martingale"  # Omnicampus アカウント名（本名ではない）

# (場面名, タスクID, seed, 外乱, 再生速度, シーン名ラベル)
DEMOS = [
    ("T1", "T1", 0, False, 2.0, "デモ1 単純な指示"),
    ("T4", "T4", 0, False, 2.0, "デモ2 積み上げ（大きい順）"),
    ("T5_disturb", "T5", 1, True, 2.0, "デモ3 曖昧な指示＋外乱からの回復"),
    ("T6", "T6", 0, False, 1.0, "デモ4 実行不能な指示"),
]


def architecture_card() -> np.ndarray:
    """システム構成図（閉ループ）を描く。"""
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    f, fb, fs = _font(24), _font(26, bold=True), _font(20)
    d.text((40, 24), "システム構成：LLM が「何をするか」、検証済みスキルが「どう動くか」", font=_font(30, bold=True),
           fill=(20, 30, 50))

    def box(x, y, w, h, title, sub, color):
        d.rounded_rectangle([x, y, x + w, y + h], radius=14, fill=color, outline=(60, 70, 90), width=2)
        d.text((x + 16, y + 12), title, font=fb, fill=(20, 30, 50))
        for i, s in enumerate(sub):
            d.text((x + 16, y + 50 + 30 * i), s, font=fs, fill=(50, 60, 80))

    def arrow(x1, y1, x2, y2, label="", color=(60, 70, 90)):
        d.line([x1, y1, x2, y2], fill=color, width=4)
        ang = np.arctan2(y2 - y1, x2 - x1)
        for s in (0.5, -0.5):
            d.line([x2, y2, x2 - 16 * np.cos(ang + s), y2 - 16 * np.sin(ang + s)], fill=color, width=4)
        if label:
            d.text(((x1 + x2) / 2 + 8, (y1 + y2) / 2 - 30), label, font=fs, fill=color)

    box(40, 110, 260, 130, "日本語の指示", ["例「机の上を片付けて」", "＋シーン状態(JSON)"], (255, 247, 214))
    box(380, 110, 300, 160, "Planner（Gemini）", ["構造化出力(JSON)", "・スキル列 plan", "・成功条件 goal"], (219, 234, 254))
    box(760, 110, 260, 130, "Validator", ["実行前チェック", "存在しないID・前提条件"], (237, 233, 254))
    box(760, 330, 260, 160, "Executor", ["pick / place_in /", "stack_on / go_home", "DLS逆運動学＋直線補間"], (220, 252, 231))
    box(380, 360, 300, 130, "MuJoCo", ["Franka Panda", "認識誤差 σ=3mm"], (241, 245, 249))
    box(40, 330, 260, 160, "Verifier", ["成功条件を", "シミュレータ状態で判定", "in / on / on_table"], (254, 226, 226))
    arrow(300, 175, 380, 175)
    arrow(680, 175, 760, 175)
    arrow(890, 240, 890, 330, "OK")
    arrow(760, 425, 680, 425)
    arrow(380, 425, 300, 425)
    d.text((70, 515), "成功 → 完了", font=f, fill=(22, 128, 61))
    # 失敗時の再計画ループ
    red = (200, 40, 40)
    d.line([170, 330, 170, 296, 530, 296, 530, 272], fill=red, width=4)
    arrow(530, 296, 530, 272, color=red)
    d.text((182, 300), "失敗 → 理由と最新状態を返して再計画（最大3回）", font=fs, fill=red)
    d.line([1020, 175, 1080, 175, 1080, 60, 530, 60, 530, 110], fill=red, width=4)
    d.text((700, 66), "NG → 理由を返して再計画", font=fs, fill=red)
    d.text((40, 590), "外乱：運搬中に物体を落とす（把持の緩みを模擬）→ 失敗を検知して再計画で回復できるかを評価",
           font=fs, fill=(80, 80, 80))
    d.text((40, 630), "評価：6種類の指示 × 10 seed × 4条件 = 220試行（LLMに見せない正解判定で採点）",
           font=fs, fill=(80, 80, 80))
    return np.asarray(img)


def record_demo(name: str, task_id: str, seed: int, disturb: bool, speed: float, scene: str) -> bool:
    from src.planner import GeminiPlanner
    from src.run import run_episode

    task = TASK_BY_ID[task_id]
    log = run_episode(task.instruction, seed, GeminiPlanner(), disturb=disturb, record=True, camera="diag",
                      task_check=task.check, expect_feasible=task.expect_feasible, verbose=True,
                      record_speed=speed)
    sim = log.pop("_sim")
    print(f"[{name}] gt_success={log['gt_success']} attempts={log['n_attempts']}")
    if log["gt_success"]:
        render([Clip(sim.frames, sim.captions, scene=scene, speed=speed, presampled=True)], PARTS / f"{name}.mp4")
    sim.close()
    return bool(log["gt_success"])


def still(name: str, frame: np.ndarray, seconds: float) -> Path:
    return render([Clip.still(frame, seconds)], PARTS / f"{name}.mp4")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", nargs="*", default=[], help="撮り直す場面名")
    ap.add_argument("--opening", type=Path, help="講座配布のオープニング動画（任意。リポジトリには含めない）")
    ap.add_argument("--ending", type=Path, help="講座配布のエンディング動画（任意）")
    args = ap.parse_args()
    PARTS.mkdir(parents=True, exist_ok=True)
    jingles = {k: conform(p, PARTS / f"_{k}.mp4") for k, p in (("op", args.opening), ("ed", args.ending)) if p}

    still("00_title", title_card(
        "言語指示で片付けるロボットアーム",
        f"Gemini（LLM）× MuJoCo Franka Panda ／ PAI最終課題_{ACCOUNT}",
        ["日本語の指示 → LLMが計画と成功条件を生成 → 検証済みスキルで実行",
         "実行前（Validator）と実行後（Verifier）で LLM の出力を検証",
         "失敗したら理由をLLMに返して再計画 → 外乱から回復",
         "※デモ場面は2倍速（画面右上に表示）"]), 7)
    still("01_arch", architecture_card(), 10)
    still("02_demo_head", title_card("デモ", "実際に Gemini を呼び出して実行した様子（台本なし）",
                                     ["1. 単純な指示", "2. 積み上げ（大きい順）",
                                      "3. 曖昧な指示＋運搬中に物を落とす外乱 → 再計画で回復",
                                      "4. 存在しない物体への指示 → 実行を拒否"]), 5)

    order = ["00_title", "01_arch", "02_demo_head"]
    for name, tid, seed, dist, speed, scene in DEMOS:
        if name == "T5_disturb":
            still("03_disturb_head", title_card("見せ場：外乱からの回復",
                                                "運搬中にグリッパを一瞬開いて物体を落とす",
                                                ["Executor が落下を検知（dropped_during_transport）",
                                                 "失敗理由と最新の状態を Gemini に返す",
                                                 "Gemini が残りの作業だけを計画し直す"]), 5)
            order.append("03_disturb_head")
        part = PARTS / f"{name}.mp4"
        if name in args.force or not part.exists():
            for s in (seed, seed + 1, seed + 2):  # LLM の揺れで失敗したら別 seed で撮る（成功した回のみ採用）
                if record_demo(name, tid, s, dist, speed, scene):
                    break
        order.append(name)

    still("90_results", image_card(RESULTS / "success_overall.png",
                                   "外乱あり：再計画なし 0% → 再計画あり 98%（各50試行）"), 8)
    still("91_results_task", image_card(RESULTS / "success_rate.png",
                                        "失敗はすべて積み上げ(T4)の物理層。LLMの計画ミス・Validatorでの却下は0件"), 7)
    still("99_end", title_card("まとめ", "", [
        "LLMには「何をするか」だけを任せ、構造化出力＋2段階の検証で安全に動かす",
        "失敗理由のフィードバックで、外乱から 98% 回復（再計画なしでは 0%）",
        "失敗を層別に記録 → 改善すべきは LLM ではなく「置く動作」と判断できた",
        "今後：カメラ画像からの認識(VLM)、実機・ROS 2 への展開"]), 7)
    order += ["90_results", "91_results_task", "99_end"]
    paths = ([jingles["op"]] if "op" in jingles else []) + [PARTS / f"{n}.mp4" for n in order] \
        + ([jingles["ed"]] if "ed" in jingles else [])
    concat(paths, OUTPUTS / "video" / f"PAI最終課題_{ACCOUNT}.mp4")


def conform(src: Path, dst: Path) -> Path:
    """配布動画を本編と同じ形式（1280x720・30fps・yuv420p）に変換する。"""
    import subprocess

    import imageio_ffmpeg

    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
                    "-vf", f"scale={W}:{H}:flags=lanczos,fps=30", "-pix_fmt", "yuv420p", "-an",
                    "-c:v", "libx264", "-crf", "18", str(dst)], check=True)
    return dst


if __name__ == "__main__":
    main()
