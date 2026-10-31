"""1 つの指示を実行する（計画 → 検証 → 実行 → 判定 → 必要なら再計画）。

使い方:
  python -m src.run --task T1 --seed 0 --record
  python -m src.run --instruction "緑のブロックを黄色の上に積んで" --seed 3
  python -m src.run --task T5 --disturb          # 運搬中に物体を落とす外乱を入れる
  python -m src.run --task T2 --no-replan        # 再計画なし（1 回だけ計画）
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src import verifier
from src.planner import GeminiPlanner, steps_to_calls
from src.sim import OUTPUTS, Sim
from src.skills import Robot
from src.tasks import TASK_BY_ID
from src.validator import validate

MAX_ATTEMPTS = 3


def fmt_step(skill: str, kw: dict) -> str:
    return f"{skill}({', '.join(v for v in kw.values() if v)})"


def run_episode(instruction: str, seed: int, planner: GeminiPlanner, *, replan: bool = True,
                disturb: bool = False, record: bool = False, camera: str = "diag",
                task_check=None, expect_feasible: bool = True, verbose: bool = True) -> dict:
    sim = Sim(seed=seed, record=record, camera=camera)
    robot = Robot(sim)
    initial_state = sim.get_state(noisy=False)
    history: list[dict] = []
    log = {"instruction": instruction, "seed": seed, "replan": replan, "disturb": disturb,
           "attempts": [], "llm_calls": 0, "input_tokens": 0, "output_tokens": 0,
           "thinking_tokens": 0, "llm_latency_s": 0.0}
    robot.slip_next = disturb
    outcome, fail_layer = "fail", ""
    max_attempts = MAX_ATTEMPTS if replan else 1
    sim.caption = f"指示: {instruction}"
    sim.hold(1.0)

    for attempt in range(1, max_attempts + 1):
        state = sim.get_state(noisy=True)
        res = planner.plan(instruction, state, history)
        log["llm_calls"] += 1
        log["input_tokens"] += res.input_tokens
        log["output_tokens"] += res.output_tokens
        log["thinking_tokens"] += res.thinking_tokens
        log["llm_latency_s"] += res.latency_s
        a = {"attempt": attempt, "error": res.error}
        log["attempts"].append(a)
        if res.output is None:
            fail_layer = "llm_error"
            history.append({"attempt": attempt, "error": res.error})
            continue
        p = res.output
        a.update(feasible=p.feasible, reasoning=p.reasoning,
                 plan=[fmt_step(*c) for c in steps_to_calls(p)],
                 goal=[f"{g.pred.value}({', '.join(g.args)})" for g in p.goal])
        if verbose:
            print(f"[attempt {attempt}] feasible={p.feasible} {p.reasoning}")
            print("   plan:", " → ".join(a["plan"]) or "(none)")
            print("   goal:", ", ".join(a["goal"]) or "(none)")

        if not p.feasible:
            outcome = "refused"
            sim.caption = f"実行不能と判断: {p.reasoning}"
            sim.hold(1.5)
            break

        problems = validate(p, state)
        if problems:
            a["validator"] = problems
            fail_layer = "validator"
            if verbose:
                print("   validator NG:", problems)
            history.append({"attempt": attempt, "plan": a["plan"], "validator_errors": problems})
            continue

        failed = None
        for i, (skill, kw) in enumerate(steps_to_calls(p)):
            sim.caption = f"[{attempt}回目] {i + 1}/{len(p.plan)} {fmt_step(skill, kw)}"
            r = robot.execute(skill, kw)
            if not r.ok:
                failed = {"step": i + 1, "skill": fmt_step(skill, kw), "reason": r.reason}
                break
        if failed:
            a["exec_failure"] = failed
            fail_layer = "execution"
            if verbose:
                print("   execution failed:", failed)
            robot.go_home()
            history.append({"attempt": attempt, "plan": a["plan"], "execution_failure": failed,
                            "note": "現在のシーン状態は最新の状態 JSON を参照"})
            continue

        unmet = [g for g in p.goal if not verifier.check(sim, g.pred.value, g.args)]
        a["unmet_goals"] = [f"{g.pred.value}({', '.join(g.args)})" for g in unmet]
        if not unmet:
            outcome = "success"
            sim.caption = "LLMの成功条件をすべて満たしました"
            sim.hold(1.0)
            break
        fail_layer = "goal_unmet"
        if verbose:
            print("   unmet goals:", a["unmet_goals"])
        history.append({"attempt": attempt, "plan": a["plan"], "unmet_goals": a["unmet_goals"]})

    # 正解判定（LLM には見せていない基準）
    final_state = sim.get_state(noisy=False)
    moved = [o["id"] for o, o0 in zip(final_state["objects"], initial_state["objects"])
             if o["location"] != o0["location"]]
    if expect_feasible:
        gt = outcome == "success" and (task_check(sim) if task_check else True)
    else:
        gt = outcome == "refused" and not moved
    log.update(outcome=outcome, gt_success=bool(gt), fail_layer="" if gt else (fail_layer or outcome),
               n_attempts=len(log["attempts"]), sim_time_s=round(sim.data.time, 1),
               final_locations={o["id"]: o["location"] for o in final_state["objects"]})
    log["_sim"] = sim
    return log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", help="T1〜T6（src/tasks.py）")
    ap.add_argument("--instruction", help="任意の指示文（--task より優先）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-replan", action="store_true")
    ap.add_argument("--disturb", action="store_true")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--camera", default="diag")
    args = ap.parse_args()

    task = TASK_BY_ID.get(args.task) if args.task else None
    instruction = args.instruction or (task.instruction if task else "赤いブロックを左の箱に入れて")
    planner = GeminiPlanner()
    t0 = time.time()
    log = run_episode(instruction, args.seed, planner, replan=not args.no_replan, disturb=args.disturb,
                      record=args.record, camera=args.camera,
                      task_check=task.check if task and not args.instruction else None,
                      expect_feasible=task.expect_feasible if task and not args.instruction else True)
    sim = log.pop("_sim")
    print(f"\n=> outcome={log['outcome']} gt_success={log['gt_success']} attempts={log['n_attempts']} "
          f"llm_calls={log['llm_calls']} tokens(in/out/think)={log['input_tokens']}/{log['output_tokens']}/"
          f"{log['thinking_tokens']} wall={time.time() - t0:.1f}s")
    print("final:", log["final_locations"])
    OUTPUTS.mkdir(exist_ok=True)
    name = f"{args.task or 'custom'}_seed{args.seed}{'_disturb' if args.disturb else ''}"
    Path(OUTPUTS / f"{name}.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.record:
        print("video:", sim.save_video(OUTPUTS / f"{name}.mp4"))
    sim.close()


if __name__ == "__main__":
    main()
