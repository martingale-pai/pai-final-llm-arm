"""一括評価：指示 × seed × 条件 を実行し、JSONL に追記する（中断しても再開できる）。

条件:
  A: 再計画なし・外乱なし（LLM の 1 発勝負の性能）
  B: 再計画あり・外乱なし
  C: 再計画あり・外乱あり（運搬中に物体を落とす）
  D: 再計画なし・外乱あり（C と比べて再計画の効果を見る）

使い方:
  python -m eval.evaluate --seeds 10                 # 実行（途中で止めても再実行で続きから）
  python -m eval.evaluate --report                   # 集計・グラフだけ作る

【学習メモ】再現可能な評価（第2回・第7回）
  - seed で初期配置を固定 → 条件間で「同じ問題」を解かせて比較できる
  - 1 試行 1 行の JSONL に生ログを残し、集計は後から何度でもやり直せるようにする
  - 「LLM の自己申告（outcome）」ではなく「正解判定（gt_success）」で成功率を出す
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sim import OUTPUTS  # noqa: E402
from src.tasks import TASKS  # noqa: E402

CONDITIONS = {
    "A": {"replan": False, "disturb": False, "label": "再計画なし"},
    "B": {"replan": True, "disturb": False, "label": "再計画あり"},
    "C": {"replan": True, "disturb": True, "label": "外乱+再計画あり"},
    "D": {"replan": False, "disturb": True, "label": "外乱+再計画なし"},
}
RESULTS = OUTPUTS / "eval" / "results.jsonl"
# 実行不能タスク（T6）はロボットが動かないので外乱条件は意味がない
SKIP = {("T6", "C"), ("T6", "D")}
# この料金は概算（2026/10 時点の公開価格を README に明記して使う）。無料枠なら実費 0
PRICE_PER_M = {"input": 0.30, "output": 2.50}


def done_keys() -> set[tuple]:
    if not RESULTS.exists():
        return set()
    keys = set()
    for line in RESULTS.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        keys.add((r["task"], r["condition"], r["seed"]))
    return keys


def run(seeds: int, tasks: list[str], conds: list[str]) -> None:
    from src.planner import GeminiPlanner
    from src.run import run_episode

    planner = GeminiPlanner()
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    done = done_keys()
    todo = [(t, c, s) for s in range(seeds) for c in conds for t in TASKS
            if t.id in tasks and (t.id, c) not in SKIP and (t.id, c, s) not in done]
    print(f"model={planner.model} 残り {len(todo)} 試行（完了済み {len(done)}）", flush=True)
    for i, (task, cond, seed) in enumerate(todo, 1):
        cfg = CONDITIONS[cond]
        t0 = time.time()
        log = run_episode(task.instruction, seed, planner, replan=cfg["replan"], disturb=cfg["disturb"],
                          task_check=task.check, expect_feasible=task.expect_feasible, verbose=False)
        log.pop("_sim").close()
        api_errors = [a["error"] for a in log["attempts"] if a.get("error")]
        # 1 日の上限など、待っても回復しない API エラーが 1 回でも出た試行は記録せずに中断（後で再開）
        # → 「API の都合で失敗」を「ロボット/LLM の失敗」として成功率に混ぜないため
        if api_errors:
            print(f"API エラーで中断します: {api_errors[-1][:200]}", flush=True)
            return
        rec = {"task": task.id, "kind": task.kind, "condition": cond, "seed": seed,
               "model": planner.model, "wall_s": round(time.time() - t0, 1), **log}
        with RESULTS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        mark = "✅" if log["gt_success"] else "❌"
        print(f"[{i}/{len(todo)}] {task.id} {cond} seed={seed} {mark} attempts={log['n_attempts']} "
              f"layer={log['fail_layer'] or '-'} {rec['wall_s']}s", flush=True)


def report() -> None:
    import matplotlib
    import numpy as np
    import pandas as pd

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # 日本語フォント（Windows 標準の Yu Gothic / Meiryo）
    for name in ("Yu Gothic", "Meiryo", "MS Gothic"):
        if any(name in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break

    rows = [json.loads(l) for l in RESULTS.read_text(encoding="utf-8").splitlines()]
    df = pd.DataFrame(rows)
    df["cost_usd"] = (df.input_tokens * PRICE_PER_M["input"]
                      + (df.output_tokens + df.thinking_tokens) * PRICE_PER_M["output"]) / 1e6
    out = OUTPUTS / "eval"
    df.drop(columns=["attempts"]).to_csv(out / "results.csv", index=False, encoding="utf-8-sig")

    task_kind = {t.id: f"{t.id} {t.kind}" for t in TASKS}
    sr = df.pivot_table(index="task", columns="condition", values="gt_success", aggfunc="mean") * 100
    n = df.pivot_table(index="task", columns="condition", values="gt_success", aggfunc="count")
    overall = df.groupby("condition").agg(
        success_rate=("gt_success", lambda x: round(100 * x.mean(), 1)), n=("gt_success", "size"),
        llm_calls=("llm_calls", "mean"), attempts=("n_attempts", "mean"),
        llm_latency_s=("llm_latency_s", "mean"), input_tokens=("input_tokens", "mean"),
        output_tokens=("output_tokens", "mean"), thinking_tokens=("thinking_tokens", "mean"),
        cost_usd=("cost_usd", "mean")).round(3)
    fails = df[~df.gt_success].groupby(["condition", "fail_layer"]).size().unstack(fill_value=0)
    # LLM の自己判定と正解判定の食い違い（LLM は成功と言ったが実は失敗）
    mismatch = df[(df.outcome == "success") & (~df.gt_success)]

    lines = ["# 評価結果", "", f"- 試行数: {len(df)}　モデル: {', '.join(df.model.unique())}", "",
             "## 条件別サマリ", "", overall.to_markdown(), "",
             "## タスク × 条件 の成功率 [%]（括弧内は試行数）", ""]
    tbl = sr.round(0).astype("Int64").astype(str) + " (" + n.astype("Int64").astype(str) + ")"
    tbl.index = [task_kind[i] for i in tbl.index]
    lines += [tbl.fillna("-").to_markdown(), "", "## 失敗の内訳（層別）", "",
              fails.to_markdown() if len(fails) else "失敗なし", "",
              f"## LLM の自己判定が成功・正解判定が失敗だった試行: {len(mismatch)} 件", ""]
    for _, r in mismatch.iterrows():
        lines.append(f"- {r.task} {r.condition} seed={r.seed}: final={r.final_locations}")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    # グラフ: タスク別 × 条件別の成功率
    fig, ax = plt.subplots(figsize=(10, 4.5))
    conds = [c for c in CONDITIONS if c in sr.columns]
    width = 0.8 / len(conds)
    colors = {"A": "#9aa5b1", "B": "#3b82f6", "C": "#16a34a", "D": "#f97316"}
    for k, c in enumerate(conds):
        xs = [i + (k - (len(conds) - 1) / 2) * width for i in range(len(sr))]
        vals = sr[c]
        ax.bar(xs, vals.fillna(0).clip(lower=1.5).where(vals.notna(), 0), width,
               label=f"{c}: {CONDITIONS[c]['label']}", color=colors[c])  # 0% も細い棒で見えるように
        for x, v in zip(xs, vals):
            if not np.isnan(v):
                ax.text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(sr)), [task_kind[i] for i in sr.index])
    ax.set_ylabel("成功率 [%]")
    ax.set_ylim(0, 112)
    ax.set_title(f"タスク別・条件別の成功率（正解判定、各 {int(n.max().max())} seed）")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "success_rate.png", dpi=150)
    print("saved:", out / "success_rate.png")

    # グラフ 2: 外乱下での再計画の効果（動画・レポートの要約用）
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [f"{c}\n{CONDITIONS[c]['label']}" for c in conds]
    vals = [overall.loc[c, "success_rate"] for c in conds]
    bars = ax.bar(labels, [max(v, 1.5) for v in vals], color=[colors[c] for c in conds])
    for b, v, c in zip(bars, vals, conds):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%\n(n={int(overall.loc[c, 'n'])})",
                ha="center", fontsize=10)
    ax.set_ylim(0, 120)
    ax.set_ylabel("成功率 [%]")
    ax.set_title("条件別の全体成功率（A・B は T1〜T6、C・D は T1〜T5 の各 10 seed）", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "success_overall.png", dpi=150)
    print("saved:", out / "success_overall.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--tasks", default=",".join(t.id for t in TASKS))
    ap.add_argument("--conditions", default="A,B,C,D")
    ap.add_argument("--report", action="store_true", help="集計とグラフのみ")
    args = ap.parse_args()
    if not args.report:
        run(args.seeds, args.tasks.split(","), args.conditions.split(","))
    if RESULTS.exists():
        report()


if __name__ == "__main__":
    main()
