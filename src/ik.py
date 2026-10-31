"""逆運動学（IK）：減衰最小二乗法（Damped Least Squares, DLS）。

手先（TCP サイト）の目標位置・姿勢 → 7 関節の角度 を反復計算で求める。

    e  = [p_target - p ; 姿勢誤差]                 (6,)
    J  = [J_pos ; J_rot]  (mj_jacSite で取得)      (6, 7)
    Δq = Jᵀ (J Jᵀ + λ² I)⁻¹ e
    q ← q + Δq   を |e| が十分小さくなるまで繰り返す

【学習メモ】λ（減衰係数）の役割
  λ=0 なら通常の擬似逆行列。特異姿勢（腕が伸び切る等）で J Jᵀ がほぼ特異になると
  Δq が巨大になり暴れる。λ² I を足すと「少し誤差が残っても関節を大きく動かさない」
  解になり安定する。λ を大きくしすぎると収束が遅くなる（トレードオフ）。
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

ARM_DOF = 7


def quat_from_yaw(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


class DLSIK:
    def __init__(self, model: mujoco.MjModel, site: str = "tcp", damping: float = 0.05,
                 max_iters: int = 100, tol_pos: float = 1e-4, tol_rot: float = 1e-3) -> None:
        self.model = model
        self.data = mujoco.MjData(model)  # 計算専用のデータ（本番の MjData は汚さない）
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        self.damping = damping
        self.max_iters = max_iters
        self.tol_pos, self.tol_rot = tol_pos, tol_rot
        self.lo = model.jnt_range[:ARM_DOF, 0].copy()
        self.hi = model.jnt_range[:ARM_DOF, 1].copy()
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def forward(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """順運動学：関節角 q → TCP の位置と姿勢（クォータニオン）。"""
        self.data.qpos[:ARM_DOF] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)  # ヤコビアン計算に必要
        pos = self.data.site_xpos[self.site].copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.site])
        return pos, quat

    def solve(self, target_pos: np.ndarray, target_quat: np.ndarray,
              q_init: np.ndarray) -> tuple[np.ndarray, float]:
        """戻り値: (関節角, 最終的な位置誤差[m])"""
        q = q_init.copy()
        lam2 = self.damping ** 2
        err_pos = np.inf
        for _ in range(self.max_iters):
            pos, quat = self.forward(q)
            e_pos = target_pos - pos
            # 姿勢誤差: quat * exp(r) = target_quat となる r（ローカル座標）→ ワールド座標へ
            r_local = np.zeros(3)
            mujoco.mju_subQuat(r_local, target_quat, quat)
            e_rot = self.data.site_xmat[self.site].reshape(3, 3) @ r_local
            err_pos = float(np.linalg.norm(e_pos))
            if err_pos < self.tol_pos and np.linalg.norm(e_rot) < self.tol_rot:
                break
            mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site)
            J = np.vstack([self._jacp[:, :ARM_DOF], self._jacr[:, :ARM_DOF]])
            e = np.concatenate([e_pos, e_rot])
            dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(6), e)
            # 1 回の更新量を制限して発散を防ぐ
            n = np.linalg.norm(dq)
            if n > 0.3:
                dq *= 0.3 / n
            q = np.clip(q + dq, self.lo, self.hi)
        return q, err_pos


def min_jerk(s: float) -> float:
    """最小躍度の時間スケーリング（0→1 を滑らかに加減速）。"""
    s = min(max(s, 0.0), 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5
