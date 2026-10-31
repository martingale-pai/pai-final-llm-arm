"""評価用の指示セットと「正解の成功判定」。

正解判定は LLM には見せない。LLM 自身が出した goal とは別に判定することで、
「LLM が間違ったゴールを立てて、それを満たして自己満足する」ケースを検出できる。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src import verifier
from src.sim import BINS, Sim


@dataclass
class Task:
    id: str
    kind: str
    instruction: str
    expect_feasible: bool
    check: Callable[[Sim], bool]


def _tower3_by_size(sim: Sim) -> bool:
    """3 段の塔があり、下から上へ大きさが小さくなる（同じ大きさは可）。"""
    blocks = list(sim.block_body)
    half = {b: sim.block_info[b]["half"] for b in blocks}
    for a in blocks:
        for b in blocks:
            for c in blocks:
                if len({a, b, c}) == 3 and verifier.on_table(sim, a) and verifier.on(sim, b, a) \
                        and verifier.on(sim, c, b) and half[a] >= half[b] >= half[c]:
                    return True
    return False


TASKS = [
    Task("T1", "単純移動", "赤いブロックを左の箱に入れて", True,
         lambda s: verifier.in_bin(s, "red_cube", "left_bin")),
    Task("T2", "属性による選択", "小さいブロックを全部右の箱に入れて", True,
         lambda s: all(verifier.in_bin(s, b, "right_bin") for b in ("blue_cube", "green_cube"))),
    Task("T3", "否定", "青以外のブロックを左の箱に入れて", True,
         lambda s: all(verifier.in_bin(s, b, "left_bin") for b in ("red_cube", "green_cube", "yellow_cube"))
         and not verifier.in_bin(s, "blue_cube", "left_bin")),
    Task("T4", "積み上げ", "ブロックを大きい順に下から3段積み上げて", True, _tower3_by_size),
    Task("T5", "曖昧な指示", "机の上を片付けて", True,
         lambda s: all(any(verifier.in_bin(s, b, k) for k in BINS) for b in s.block_body)),
    Task("T6", "実行不能", "紫のブロックを左の箱に入れて", False,
         lambda s: True),  # 実行不能と判断して何も動かさなければ成功（run 側で判定）
]
TASK_BY_ID = {t.id: t for t in TASKS}
