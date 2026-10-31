"""動画編集：字幕の焼き込み・タイトルカード・画像カードをつないで 1 本の mp4 にする。

シミュレーションの録画フレーム（sim.frames）と、各フレーム時点の字幕（sim.captions）を受け取り、
画面下部に字幕帯、左上にシーン名、右上に再生速度を描画する。

【学習メモ】なぜ字幕を「後から」焼き込むのか
  シミュレーション中は生フレームと字幕テキストだけを記録し、描画は最後にまとめて行う。
  こうしておくと、字幕のデザインを変えてもシミュレーション（と LLM 呼び出し）をやり直さずに済む。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_REGULAR = "C:/Windows/Fonts/YuGothM.ttc"
FONT_BOLD = "C:/Windows/Fonts/YuGothB.ttc"
W, H = 1280, 720
FPS = 30


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)
    except OSError:  # Windows 以外の環境では Noto CJK などを探す
        for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                  "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc"):
            if Path(p).exists():
                return ImageFont.truetype(p, size)
        return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_w: int) -> list[str]:
    """日本語は単語区切りがないので 1 文字ずつ幅を測って折り返す。"""
    lines = []
    for para in text.split("\n"):
        cur = ""
        for ch in para:
            if draw.textlength(cur + ch, font=font) > max_w and cur:
                lines.append(cur)
                cur = ch
            else:
                cur += ch
        lines.append(cur)
    return lines


def overlay(frame: np.ndarray, caption: str, scene: str = "", speed: str = "") -> np.ndarray:
    img = Image.fromarray(frame).convert("RGBA")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f_cap, f_tag = _font(28), _font(24, bold=True)
    if caption:
        lines = _wrap(d, caption, f_cap, W - 80)[:4]
        bar_h = 24 + 40 * len(lines)
        d.rectangle([0, H - bar_h, W, H], fill=(0, 0, 0, 170))
        for i, line in enumerate(lines):
            color = (255, 230, 120, 255) if i == 0 else (255, 255, 255, 255)
            d.text((40, H - bar_h + 12 + 40 * i), line, font=f_cap, fill=color)
    for text, right in ((scene, False), (speed, True)):
        if not text:
            continue
        tw = d.textlength(text, font=f_tag)
        x = W - tw - 40 if right else 20
        d.rounded_rectangle([x - 10, 16, x + tw + 10, 56], radius=8, fill=(20, 60, 140, 210))
        d.text((x, 22), text, font=f_tag, fill=(255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(img, layer).convert("RGB"))


def title_card(title: str, subtitle: str = "", bullets: list[str] | None = None) -> np.ndarray:
    img = Image.new("RGB", (W, H), (18, 24, 38))
    d = ImageDraw.Draw(img)
    y = 150 if bullets else 280
    for line in _wrap(d, title, _font(52, bold=True), W - 160):
        d.text((80, y), line, font=_font(52, bold=True), fill=(255, 255, 255))
        y += 70
    if subtitle:
        for line in _wrap(d, subtitle, _font(30), W - 160):
            d.text((80, y + 10), line, font=_font(30), fill=(150, 200, 255))
            y += 44
    y += 40
    for b in bullets or []:
        for j, line in enumerate(_wrap(d, b, _font(30), W - 220)):
            d.text((100, y), ("・" if j == 0 else "　") + line, font=_font(30), fill=(225, 230, 240))
            y += 46
    return np.asarray(img)


def image_card(path: Path, caption: str = "") -> np.ndarray:
    """グラフや構成図などの画像を 16:9 の画面に収める。"""
    bg = Image.new("RGB", (W, H), (255, 255, 255))
    im = Image.open(path).convert("RGB")
    max_h = H - (90 if caption else 20)
    im.thumbnail((W - 40, max_h))
    bg.paste(im, ((W - im.width) // 2, 10 + (max_h - im.height) // 2))
    frame = np.asarray(bg)
    return overlay(frame, caption) if caption else frame


@dataclass
class Clip:
    """1 つの場面。frames と captions は同じ長さ。"""
    frames: list[np.ndarray]
    captions: list[str] = field(default_factory=list)
    scene: str = ""
    speed: float = 1.0  # 2.0 なら 1 フレームおきに間引いて 2 倍速にする

    @classmethod
    def still(cls, frame: np.ndarray, seconds: float) -> "Clip":
        return cls([frame] * int(seconds * FPS), [""] * int(seconds * FPS))


def render(clips: list[Clip], path: Path) -> Path:
    out: list[np.ndarray] = []
    for c in clips:
        step = max(1, int(round(c.speed)))
        caps = c.captions or [""] * len(c.frames)
        speed_tag = f"{c.speed:g}倍速" if c.speed != 1 else ""
        for f, cap in zip(c.frames[::step], caps[::step]):
            out.append(overlay(f, cap, c.scene, speed_tag) if (cap or c.scene or speed_tag) else f)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, np.stack(out), fps=FPS, codec="libx264", quality=8)
    print(f"saved: {path}  ({len(out) / FPS:.1f} 秒)")
    return path
