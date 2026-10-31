"""シミュレータの状態から述語（in / on / on_table / holding）を判定する。

LLM が出した成功条件（goal）も、評価用の正解ゴールも、ここで同じ基準で判定する。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.sim import BIN_HALF, BINS, TABLE_Z

if TYPE_CHECKING:
    from src.sim import Sim


def in_bin(sim: "Sim", obj: str, bin_id: str) -> bool:
    p = sim.object_pose(obj).pos
    c = BINS[bin_id]
    inside_xy = np.all(np.abs(p[:2] - c) < BIN_HALF)
    resting = p[2] < TABLE_Z + 0.09  # 箱の中にある（持ち上げられていない）
    return bool(inside_xy and resting)


def on(sim: "Sim", obj: str, base: str) -> bool:
    """obj が base の上に載っている。"""
    po, pb = sim.object_pose(obj), sim.object_pose(base)
    xy_ok = np.linalg.norm(po.pos[:2] - pb.pos[:2]) < pb.half * 0.9
    z_expected = pb.pos[2] + pb.half + po.half
    z_ok = abs(po.pos[2] - z_expected) < 0.012
    return bool(xy_ok and z_ok)


def holding(sim: "Sim", obj: str) -> bool:
    """obj がグリッパに把持されている（手先の近くにあり、テーブルから浮いている）。"""
    p = sim.object_pose(obj)
    near_tcp = np.linalg.norm(p.pos - sim.tcp_pos()) < 0.03
    lifted = p.pos[2] > TABLE_Z + p.half + 0.01
    return bool(near_tcp and lifted and sim.gripper_width() > 0.005)


def on_table(sim: "Sim", obj: str) -> bool:
    p = sim.object_pose(obj)
    if any(in_bin(sim, obj, b) for b in BINS):
        return False
    return bool(abs(p.pos[2] - (TABLE_Z + p.half)) < 0.008)


def describe_location(sim: "Sim", obj: str) -> str:
    """状態 JSON 用の位置の説明: table / left_bin / right_bin / on:<id> / gripper / unknown"""
    if holding(sim, obj):
        return "gripper"
    for b in BINS:
        if in_bin(sim, obj, b):
            return b
    for other in sim.block_body:
        if other != obj and on(sim, obj, other):
            return f"on:{other}"
    if on_table(sim, obj):
        return "table"
    return "unknown"


PREDICATES = {"in": in_bin, "on": on, "holding": holding, "on_table": on_table}


def check(sim: "Sim", pred: str, args: list[str]) -> bool:
    return PREDICATES[pred](sim, *args)
