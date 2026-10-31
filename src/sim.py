"""MuJoCo シミュレーション環境。

- Panda（mujoco_menagerie）を読み込み、MjSpec でテーブル・ブロック・箱・カメラを追加する
- seed からブロックの初期配置をランダム生成する（再現可能な評価のため）
- LLM に渡す状態 JSON の生成、外乱の注入、録画を担当する

【学習メモ】MjSpec
  XML を手で書く代わりに、Python からモデルを組み立てる API（MuJoCo 3.2+）。
  ここでは「既存のロボット XML を読み込み → 物体を足す → compile」という使い方をしている。
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
OUTPUTS = ROOT / "outputs"
PANDA_SCENE = ASSETS / "franka_emika_panda" / "scene.xml"

TABLE_Z = 0.40  # テーブル上面の高さ [m]
HOME_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
GRIPPER_OPEN, GRIPPER_CLOSE = 255.0, 0.0  # actuator8 の指令値（255 で全開 0.08m）

# ブロック: (ID, 色, 大きさ)
BLOCKS = [
    ("red_cube", "red", "large"),
    ("blue_cube", "blue", "small"),
    ("green_cube", "green", "small"),
    ("yellow_cube", "yellow", "large"),
]
RGBA = {
    "red": [0.85, 0.15, 0.15, 1],
    "blue": [0.15, 0.35, 0.85, 1],
    "green": [0.15, 0.7, 0.25, 1],
    "yellow": [0.95, 0.8, 0.1, 1],
}
HALF = {"small": 0.020, "large": 0.025}  # 立方体の半辺長 [m]

# 箱: ロボットから見て左（+y）と右（-y）
BINS = {"left_bin": np.array([0.45, 0.36]), "right_bin": np.array([0.45, -0.36])}
BIN_HALF = 0.075  # 内寸の半分 [m]
BIN_WALL_H = 0.04
BIN_WALL_T = 0.006
BIN_FLOOR_T = 0.004

# ブロックを置く領域（テーブル中央）
SPAWN_X = (0.40, 0.62)
SPAWN_Y = (-0.18, 0.18)
SPAWN_MIN_DIST = 0.10

CTRL_DT = 0.02  # 制御周期 50Hz（物理は 2ms × 10 ステップ）
VIDEO_FPS = 30


def _build_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(PANDA_SCENE))
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    # menagerie の keyframe は Panda 単体の関節数（9）用なので、物体を足すと不整合になる → 削除
    for key in list(spec.keys):
        spec.delete(key)

    # ロボットをテーブルの上に載せる
    spec.body("link0").pos = [0.0, 0.0, TABLE_Z]
    # 手先の基準点（TCP: Tool Center Point）。指パッドの中心付近
    spec.body("hand").add_site(name="tcp", pos=[0, 0, 0.1034], size=[0.004] * 3, rgba=[1, 0, 0, 0])

    wb = spec.worldbody
    box = mujoco.mjtGeom.mjGEOM_BOX
    wb.add_geom(name="table", type=box, size=[0.6, 0.65, TABLE_Z / 2], pos=[0.3, 0, TABLE_Z / 2],
                rgba=[0.62, 0.5, 0.38, 1])

    # 箱（床板 + 4枚の壁）。固定物なので worldbody に直接 geom を置く
    for name, (bx, by) in BINS.items():
        rgba = [0.85, 0.85, 0.85, 1]
        z0 = TABLE_Z + BIN_FLOOR_T / 2
        wb.add_geom(name=f"{name}_floor", type=box, size=[BIN_HALF, BIN_HALF, BIN_FLOOR_T / 2],
                    pos=[bx, by, z0], rgba=rgba)
        zw = TABLE_Z + BIN_WALL_H / 2
        off = BIN_HALF + BIN_WALL_T / 2
        long = BIN_HALF + BIN_WALL_T
        for i, (dx, dy, sx, sy) in enumerate([(off, 0, BIN_WALL_T / 2, long), (-off, 0, BIN_WALL_T / 2, long),
                                              (0, off, long, BIN_WALL_T / 2), (0, -off, long, BIN_WALL_T / 2)]):
            wb.add_geom(name=f"{name}_wall{i}", type=box, size=[sx, sy, BIN_WALL_H / 2],
                        pos=[bx + dx, by + dy, zw], rgba=rgba)

    # ブロック（自由関節 = 6自由度で動ける物体）
    for bid, color, size in BLOCKS:
        h = HALF[size]
        body = wb.add_body(name=bid, pos=[0.5, 0, TABLE_Z + h])
        body.add_freejoint()
        body.add_geom(name=bid, type=box, size=[h, h, h], rgba=RGBA[color], mass=0.05,
                      friction=[1.2, 0.02, 0.002], condim=4)

    # カメラ（注視点のダミー body を向くモード）
    wb.add_body(name="ws_center", pos=[0.48, 0, TABLE_Z])
    # front: ロボットと向かい合う視点（画面の左右とロボットの左右は逆になる点に注意）
    # diag: 斜め前からの見栄えの良い視点（動画用）
    for cname, pos, fovy in [("front", [1.25, 0.0, 1.05], 50), ("side", [0.55, -1.05, 0.95], 50),
                             ("diag", [1.05, -0.75, 1.0], 50)]:
        cam = wb.add_camera(name=cname, pos=pos, fovy=fovy)
        cam.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
        cam.targetbody = "ws_center"
    return spec


_MODEL_CACHE: mujoco.MjModel | None = None


def build_model() -> mujoco.MjModel:
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = _build_spec().compile()
    return _MODEL_CACHE


def wrap_yaw(yaw: float) -> float:
    """立方体は 90° ごとに同じ形なので、yaw を [-45°, 45°) に折り返す。"""
    return (yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4


@dataclass
class ObjectPose:
    pos: np.ndarray
    yaw: float
    half: float


class Sim:
    """シーンの生成・状態取得・外乱・録画をまとめたクラス。"""

    def __init__(self, seed: int = 0, record: bool = False, obs_noise: float = 0.003,
                 camera: str = "front", width: int = 1280, height: int = 720,
                 record_speed: float = 1.0) -> None:
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        self.obs_noise = obs_noise
        self.record = record
        self.camera = camera
        self.frames: list[np.ndarray] = []
        self.captions: list[str] = []
        self.caption = ""
        self._renderer = mujoco.Renderer(self.model, height=height, width=width) if record else None
        self._next_frame_t = 0.0
        # 2 倍速で使う動画なら、最初から 2 倍の間隔でフレームを取る（メモリ節約）
        self.frame_dt = record_speed / VIDEO_FPS
        m = self.model
        self.tcp_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.block_body = {b[0]: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b[0]) for b in BLOCKS}
        self.block_qadr = {b: m.jnt_qposadr[m.body_jntadr[i]] for b, i in self.block_body.items()}
        self.block_info = {b[0]: {"color": b[1], "size": b[2], "half": HALF[b[2]]} for b in BLOCKS}
        self.finger_qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
                            for j in ("finger_joint1", "finger_joint2")]
        self.reset(seed)

    # ------------------------------------------------------------------ 初期化
    def reset(self, seed: int) -> None:
        """seed から初期配置を決める。同じ seed なら必ず同じ配置になる。"""
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[:7] = HOME_QPOS
        d.qpos[self.finger_qadr] = 0.04
        d.ctrl[:7] = HOME_QPOS
        d.ctrl[7] = GRIPPER_OPEN

        placed: list[np.ndarray] = []
        for bid in self.block_body:
            for _ in range(1000):
                xy = np.array([self.rng.uniform(*SPAWN_X), self.rng.uniform(*SPAWN_Y)])
                if all(np.linalg.norm(xy - p) > SPAWN_MIN_DIST for p in placed):
                    break
            placed.append(xy)
            yaw = self.rng.uniform(-math.pi / 4, math.pi / 4)
            a = self.block_qadr[bid]
            d.qpos[a:a + 3] = [xy[0], xy[1], TABLE_Z + self.block_info[bid]["half"]]
            d.qpos[a + 3:a + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        mujoco.mj_forward(m, d)
        self.frames.clear()
        self.captions.clear()
        self._next_frame_t = 0.0
        self.step(100)  # 物体を接地させる

    # ------------------------------------------------------------------ 物理
    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            if self.record and self.data.time >= self._next_frame_t:
                self._capture()
                self._next_frame_t += self.frame_dt

    def _capture(self) -> None:
        self._renderer.update_scene(self.data, camera=self.camera)
        self.frames.append(self._renderer.render().copy())
        self.captions.append(self.caption)

    # ------------------------------------------------------------------ 観測
    def tcp_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.tcp_site].copy()

    def gripper_width(self) -> float:
        return float(self.data.qpos[self.finger_qadr].sum())

    def object_pose(self, bid: str, noisy: bool = False) -> ObjectPose:
        bi = self.block_body[bid]
        pos = self.data.xpos[bi].copy()
        R = self.data.xmat[bi].reshape(3, 3)
        yaw = wrap_yaw(math.atan2(R[1, 0], R[0, 0]))
        if noisy and self.obs_noise > 0:
            pos[:2] += self.rng.normal(0, self.obs_noise, 2)
        return ObjectPose(pos, yaw, self.block_info[bid]["half"])

    def get_state(self, noisy: bool = True) -> dict:
        """LLM に渡すシーン状態。位置は mm 単位に丸める（LLM には過剰な桁は不要）。"""
        from src.verifier import describe_location  # 循環 import 回避

        objects = []
        for bid, info in self.block_info.items():
            p = self.object_pose(bid, noisy=noisy)
            objects.append({
                "id": bid, "color": info["color"], "size": info["size"],
                "size_cm": round(info["half"] * 200, 1),
                "pos_m": [round(float(v), 3) for v in p.pos],
                "location": describe_location(self, bid),
            })
        bins = [{"id": k, "center_m": [round(float(v), 3) for v in xy], "side": k.split("_")[0]}
                for k, xy in BINS.items()]
        holding = next((o["id"] for o in objects if o["location"] == "gripper"), None)
        return {"objects": objects, "bins": bins, "gripper": {"holding": holding}}

    # ------------------------------------------------------------------ 外乱・録画
    def disturb(self, bid: str, force_xy: tuple[float, float], duration: float = 0.05) -> None:
        """物体に水平方向の力を一瞬加える（人がぶつかった・物が滑った状況の模擬）。"""
        bi = self.block_body[bid]
        self.data.xfrc_applied[bi, :2] = force_xy
        self.step(int(duration / self.model.opt.timestep))
        self.data.xfrc_applied[bi, :] = 0

    def hold(self, seconds: float) -> None:
        """現在の指令を保ったまま時間を進める（録画の間を作るのにも使う）。"""
        self.step(int(seconds / self.model.opt.timestep))

    def snapshot(self, camera: str | None = None) -> np.ndarray:
        r = self._renderer or mujoco.Renderer(self.model, height=720, width=1280)
        r.update_scene(self.data, camera=camera or self.camera)
        return r.render().copy()

    def save_video(self, path: Path, speed: float = 1.0) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        frames = self.frames[::max(1, int(round(speed)))]
        iio.imwrite(path, np.stack(frames), fps=VIDEO_FPS, codec="libx264", quality=8)
        return path

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def main() -> None:
    parser = argparse.ArgumentParser(description="シーンの動作確認")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--snapshot", action="store_true", help="front / side カメラの静止画を保存")
    args = parser.parse_args()

    sim = Sim(seed=args.seed)
    m = sim.model
    print(f"nq={m.nq} nv={m.nv} nu={m.nu} timestep={m.opt.timestep}")
    import json
    print(json.dumps(sim.get_state(noisy=False), ensure_ascii=False, indent=1))
    if args.snapshot:
        OUTPUTS.mkdir(exist_ok=True)
        for cam in ("front", "side"):
            path = OUTPUTS / f"scene_seed{args.seed}_{cam}.png"
            iio.imwrite(path, sim.snapshot(cam))
            print(f"saved: {path}")


if __name__ == "__main__":
    main()
