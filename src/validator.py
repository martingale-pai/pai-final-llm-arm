"""実行前の計画チェック（Validator）。

ロボットを動かす前に、計画を「記号的に」シミュレーションして明らかな誤りを見つける。
例: 存在しない物体ID / 持っていないのに置く / 上に物が載っている物体を掴む。

【学習メモ】なぜ実行前にチェックするのか
  実機ではロボットを動かすこと自体にコスト（時間・安全・物の破損）がある。
  安く検出できる誤りは動かす前に弾き、LLM に理由を返して直させる方が効率が良い。
  ProgPrompt の assertion や、古典的プランニングの前提条件チェックと同じ発想。
"""
from __future__ import annotations

from src.planner import PlanOutput

BIN_IDS = {"left_bin", "right_bin"}


def validate(plan: PlanOutput, state: dict) -> list[str]:
    """問題点のリストを返す（空なら OK）。"""
    objs = {o["id"] for o in state["objects"]}
    # 記号的な状態: 各物体の場所と、何の上に何が載っているか
    loc = {o["id"]: o["location"] for o in state["objects"]}
    holding = state["gripper"]["holding"]
    errors: list[str] = []

    def on_top_of(x: str) -> list[str]:
        return [o for o, l in loc.items() if l == f"on:{x}"]

    for i, s in enumerate(plan.plan):
        tag = f"step{i + 1} {s.skill.value}"
        if s.skill.value in ("pick", "place_in", "stack_on") and s.obj not in objs:
            errors.append(f"{tag}: unknown object '{s.obj}'")
            continue
        if s.skill.value == "pick":
            if holding:
                errors.append(f"{tag}: already holding {holding}")
            elif on_top_of(s.obj):
                errors.append(f"{tag}: {on_top_of(s.obj)[0]} is on top of {s.obj}")
            holding, loc[s.obj] = s.obj, "gripper"
        elif s.skill.value == "place_in":
            if s.bin_id not in BIN_IDS:
                errors.append(f"{tag}: unknown bin '{s.bin_id}'")
            if holding != s.obj:
                errors.append(f"{tag}: not holding {s.obj} (holding {holding})")
            holding, loc[s.obj] = None, s.bin_id or "?"
        elif s.skill.value == "stack_on":
            if s.base not in objs or s.base == s.obj:
                errors.append(f"{tag}: invalid base '{s.base}'")
            elif on_top_of(s.base):
                errors.append(f"{tag}: {on_top_of(s.base)[0]} is already on {s.base}")
            elif loc.get(s.base) == "gripper":
                errors.append(f"{tag}: base {s.base} is being held")
            if holding != s.obj:
                errors.append(f"{tag}: not holding {s.obj} (holding {holding})")
            holding, loc[s.obj] = None, f"on:{s.base}"
    if holding:
        errors.append(f"plan ends while still holding {holding}")

    for g in plan.goal:
        arity = {"in": 2, "on": 2, "on_table": 1}[g.pred.value]
        if len(g.args) != arity:
            errors.append(f"goal {g.pred.value}{g.args}: needs {arity} args")
            continue
        valid = objs | (BIN_IDS if g.pred.value == "in" else set())
        bad = [a for a in g.args if a not in valid]
        if bad:
            errors.append(f"goal {g.pred.value}{g.args}: unknown id {bad}")
    if plan.feasible and not plan.goal:
        errors.append("feasible plan must have at least one goal")
    return errors
