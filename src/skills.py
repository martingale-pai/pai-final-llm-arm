"""ロボットのスキル（検証済みの基本動作）。

LLM は「どのスキルをどの引数で呼ぶか（What）」だけを決め、
「どう動くか（How）」はこのファイルのスキルが担う。

各スキルは SkillResult(ok, reason) を返す。失敗理由は再計画時に LLM へそのまま渡すので、
人間にも LLM にも分かる短い英語＋具体的な対象名で書く。
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

from src import verifier
from src.ik import DLSIK, min_jerk, quat_from_yaw, quat_mul
from src.sim import (BIN_FLOOR_T, BINS, CTRL_DT, GRIPPER_CLOSE, GRIPPER_OPEN, HOME_QPOS, OUTPUTS,
                     TABLE_Z, Sim)

SAFE_Z = TABLE_Z + 0.22   # 物体を運ぶときの高さ
APPROACH_DZ = 0.10        # 掴む・置く直前に待機する高さ（物体上面からの距離）
SPEED_FAST, SPEED_SLOW = 0.30, 0.10  # [m/s]


@dataclass
class SkillResult:
    ok: bool
    reason: str = ""


class Robot:
    def __init__(self, sim: Sim) -> None:
        self.sim = sim
        self.ik = DLSIK(sim.model)
        _, self.down_quat = self.ik.forward(HOME_QPOS)  # home 姿勢 = グリッパが真下を向く
        self.q_cmd = HOME_QPOS.copy()
        self.yaw = 0.0
        self.pos_cmd = self.ik.forward(HOME_QPOS)[0]
        self.slip_next = False  # True にすると次の運搬中に物体を落とす（外乱評価用）

    # ------------------------------------------------------------------ 低レベル動作
    def _steps_per_tick(self) -> int:
        return int(round(CTRL_DT / self.sim.model.opt.timestep))

    def move_linear(self, target: np.ndarray, yaw: float | None = None,
                    speed: float = SPEED_FAST) -> SkillResult:
        """手先を直線で target まで動かす（各制御周期で IK を解く）。"""
        yaw = self.yaw if yaw is None else yaw
        p0, y0 = self.pos_cmd.copy(), self.yaw
        dist = float(np.linalg.norm(target - p0))
        duration = max(dist / speed, abs(yaw - y0) / 1.5, 0.3)
        ticks = max(1, int(duration / CTRL_DT))
        for k in range(1, ticks + 1):
            s = min_jerk(k / ticks)
            p = p0 + (target - p0) * s
            y = y0 + (yaw - y0) * s
            q, err = self.ik.solve(p, quat_mul(quat_from_yaw(y), self.down_quat), self.q_cmd)
            if err > 0.01:
                return SkillResult(False, f"ik_unreachable: target {np.round(target, 3).tolist()}")
            self.q_cmd = q
            self.sim.data.ctrl[:7] = q
            self.sim.step(self._steps_per_tick())
        self.pos_cmd, self.yaw = target.copy(), yaw
        # 実機と同じく、指令値に実際の手先が追いつくまで少し待つ
        for _ in range(int(0.6 / CTRL_DT)):
            if np.linalg.norm(self.sim.tcp_pos() - target) < 0.004:
                break
            self.sim.step(self._steps_per_tick())
        return SkillResult(True)

    def gripper(self, open_: bool, wait: float = 0.5) -> None:
        self.sim.data.ctrl[7] = GRIPPER_OPEN if open_ else GRIPPER_CLOSE
        self.sim.hold(wait)

    def held_object(self) -> str | None:
        return next((b for b in self.sim.block_body if verifier.holding(self.sim, b)), None)

    def _blocked_by(self, obj: str) -> str | None:
        return next((o for o in self.sim.block_body if o != obj and verifier.on(self.sim, o, obj)), None)

    # ------------------------------------------------------------------ スキル
    def go_home(self) -> SkillResult:
        q0 = self.q_cmd.copy()
        ticks = int(1.5 / CTRL_DT)
        for k in range(1, ticks + 1):
            self.q_cmd = q0 + (HOME_QPOS - q0) * min_jerk(k / ticks)
            self.sim.data.ctrl[:7] = self.q_cmd
            self.sim.step(self._steps_per_tick())
        self.pos_cmd, self.yaw = self.ik.forward(HOME_QPOS)[0], 0.0
        return SkillResult(True)

    def pick(self, obj: str) -> SkillResult:
        if obj not in self.sim.block_body:
            return SkillResult(False, f"unknown_object: {obj}")
        held = self.held_object()
        if held:
            return SkillResult(False, f"already_holding: {held}")
        blocker = self._blocked_by(obj)
        if blocker:
            return SkillResult(False, f"object_blocked: {blocker} is on top of {obj}")

        pose = self.sim.object_pose(obj, noisy=True)  # 認識誤差つきの観測で動く
        grasp = pose.pos.copy()
        above = grasp + [0, 0, pose.half + APPROACH_DZ]
        self.gripper(open_=True, wait=0.2)
        for target, speed in [(np.array([*above[:2], max(above[2], SAFE_Z)]), SPEED_FAST),
                              (above, SPEED_FAST), (grasp, SPEED_SLOW)]:
            r = self.move_linear(target, yaw=pose.yaw, speed=speed)
            if not r.ok:
                return r
        self.gripper(open_=False, wait=0.6)
        r = self.move_linear(np.array([*grasp[:2], SAFE_Z]), speed=SPEED_SLOW)
        if not r.ok:
            return r
        if not verifier.holding(self.sim, obj):
            return SkillResult(False, f"grasp_failed: {obj} slipped or was not between the fingers")
        return SkillResult(True)

    def _transport(self, obj: str, xy: np.ndarray, yaw: float) -> SkillResult:
        if self.slip_next:
            # 外乱の模擬: 運搬の途中で把持が緩み、物体を落とす
            self.slip_next = False
            mid = (self.pos_cmd[:2] + xy) / 2
            self.move_linear(np.array([mid[0], mid[1], SAFE_Z]), yaw=yaw)
            self.gripper(open_=True, wait=0.4)
            self.gripper(open_=False, wait=0.2)
        r = self.move_linear(np.array([xy[0], xy[1], SAFE_Z]), yaw=yaw)
        if r.ok and not verifier.holding(self.sim, obj):
            return SkillResult(False, f"dropped_during_transport: {obj}")
        return r

    def _release_and_retreat(self, xy: np.ndarray, release_z: float) -> SkillResult:
        r = self.move_linear(np.array([xy[0], xy[1], release_z]), speed=SPEED_SLOW)
        if not r.ok:
            return r
        self.gripper(open_=True, wait=0.5)
        return self.move_linear(np.array([xy[0], xy[1], SAFE_Z]), speed=SPEED_SLOW)

    def place_in(self, obj: str, bin_id: str) -> SkillResult:
        if bin_id not in BINS:
            return SkillResult(False, f"unknown_bin: {bin_id}")
        if not verifier.holding(self.sim, obj):
            return SkillResult(False, f"not_holding: {obj}")
        # 箱の中の空いている場所（4 区画のうち既存の物体から一番遠い所）を選ぶ
        c = BINS[bin_id]
        slots = [c + np.array([dx, dy]) for dx in (-0.035, 0.035) for dy in (-0.035, 0.035)]
        others = [self.sim.object_pose(o).pos[:2] for o in self.sim.block_body
                  if o != obj and verifier.in_bin(self.sim, o, bin_id)]
        xy = max(slots, key=lambda s: min([np.linalg.norm(s - p) for p in others] or [1.0]))
        half = self.sim.block_info[obj]["half"]
        r = self._transport(obj, xy, yaw=0.0)
        if r.ok:
            r = self._release_and_retreat(xy, TABLE_Z + BIN_FLOOR_T + half + 0.012)
        if r.ok and not verifier.in_bin(self.sim, obj, bin_id):
            return SkillResult(False, f"place_failed: {obj} is not inside {bin_id}")
        return r

    def stack_on(self, obj: str, base: str) -> SkillResult:
        if base not in self.sim.block_body or base == obj:
            return SkillResult(False, f"invalid_base: {base}")
        if not verifier.holding(self.sim, obj):
            return SkillResult(False, f"not_holding: {obj}")
        blocker = self._blocked_by(base)
        if blocker:
            return SkillResult(False, f"base_occupied: {blocker} is already on {base}")
        bp = self.sim.object_pose(base, noisy=True)
        half = self.sim.block_info[obj]["half"]
        r = self._transport(obj, bp.pos[:2], yaw=bp.yaw)
        if r.ok:
            r = self._release_and_retreat(bp.pos[:2], bp.pos[2] + bp.half + half + 0.006)
        if r.ok and not verifier.on(self.sim, obj, base):
            return SkillResult(False, f"stack_failed: {obj} is not on {base}")
        return r

    def execute(self, skill: str, args: dict) -> SkillResult:
        """LLM の計画（skill 名 + 引数 dict）からスキルを呼び出す入口。"""
        fn = {"pick": self.pick, "place_in": self.place_in, "stack_on": self.stack_on,
              "go_home": self.go_home}.get(skill)
        if fn is None:
            return SkillResult(False, f"unknown_skill: {skill}")
        try:
            return fn(**args)
        except TypeError as e:
            return SkillResult(False, f"bad_arguments for {skill}: {e}")


# ---------------------------------------------------------------------- 動作確認
DEMO_PLAN = [
    ("pick", {"obj": "red_cube"}),
    ("place_in", {"obj": "red_cube", "bin_id": "left_bin"}),
    ("pick", {"obj": "blue_cube"}),
    ("stack_on", {"obj": "blue_cube", "base": "yellow_cube"}),
    ("pick", {"obj": "green_cube"}),
    ("place_in", {"obj": "green_cube", "bin_id": "right_bin"}),
    ("go_home", {}),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM なしで手書きの計画を実行してスキルを検証する")
    parser.add_argument("--trials", type=int, default=1, help="seed 0..N-1 で繰り返す")
    parser.add_argument("--record", action="store_true", help="seed 0 の様子を mp4 で保存")
    parser.add_argument("--camera", default="diag")
    args = parser.parse_args()

    stats: dict[str, list[bool]] = {}
    for seed in range(args.trials):
        sim = Sim(seed=seed, record=args.record and seed == 0, camera=args.camera)
        robot = Robot(sim)
        t0 = time.time()
        log = []
        for skill, kw in DEMO_PLAN:
            sim.caption = f"{skill}({', '.join(kw.values())})"
            r = robot.execute(skill, kw)
            stats.setdefault(skill, []).append(r.ok)
            log.append(f"{skill}{'' if r.ok else ' FAIL ' + r.reason}")
            if not r.ok:
                break
        ok = all(s.split()[0] == s for s in log) and len(log) == len(DEMO_PLAN)
        print(f"seed={seed} {'SUCCESS' if ok else 'FAIL'} ({time.time() - t0:.1f}s) {' | '.join(log)}")
        if sim.record:
            print("video:", sim.save_video(OUTPUTS / "step2_demo.mp4"))
        sim.close()
    print("\nスキル別成功率:")
    for skill, res in stats.items():
        print(f"  {skill:9s} {sum(res)}/{len(res)}")


if __name__ == "__main__":
    main()
