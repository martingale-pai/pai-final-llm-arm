"""LLM プランナ（Gemini）。

指示文とシーン状態から、スキル列（plan）と成功条件（goal）を JSON で生成する。

【学習メモ】構造化出力（Structured Output）
  response_schema に Pydantic モデルを渡すと、Gemini は必ずそのスキーマに沿った JSON を返す。
  自由文をパースする方式より壊れにくく、存在しないスキル名なども enum で防げる。
  ただし「スキーマに合っている」≠「正しい計画」なので、Validator / Verifier で中身を検証する。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
# 無料枠での実測（2026/10/31）: 3.5-flash ≈8s, 3.5-flash-lite ≈2s, 3.8-flash ≈160s（待ち行列）
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")


# ---------------------------------------------------------------------- 出力スキーマ
class SkillName(str, Enum):
    pick = "pick"
    place_in = "place_in"
    stack_on = "stack_on"
    go_home = "go_home"


class Step(BaseModel):
    skill: SkillName
    obj: str | None = Field(None, description="pick / place_in / stack_on で動かす物体ID")
    bin_id: str | None = Field(None, description="place_in の箱ID")
    base: str | None = Field(None, description="stack_on の土台にする物体ID")


class PredName(str, Enum):
    in_ = "in"
    on = "on"
    on_table = "on_table"


class Goal(BaseModel):
    pred: PredName
    args: list[str] = Field(description="in: [物体ID, 箱ID] / on: [上の物体ID, 下の物体ID] / on_table: [物体ID]")


class PlanOutput(BaseModel):
    feasible: bool = Field(description="指示が実行可能なら true。対象が存在しない等なら false")
    reasoning: str = Field(description="指示の解釈と計画の根拠（日本語・2〜3文）")
    plan: list[Step]
    goal: list[Goal] = Field(description="タスク完了時に成り立つべき条件。指示に関係する物体すべてについて書く")


SYSTEM_PROMPT = """\
あなたはロボットアーム（Franka Panda）のタスクプランナです。
ユーザの日本語の指示とシーン状態(JSON)から、実行するスキル列と成功条件を出力してください。

# 座標系
- ロボットは +x 方向を向いている。ロボットから見て「左」= +y 側 = left_bin、「右」= -y 側 = right_bin。
- 「手前」= x が小さい（ロボットに近い）、「奥」= x が大きい。

# 使えるスキル（これ以外は使えない）
- pick(obj): 物体を掴んで持ち上げる。前提: 何も持っていない / obj の上に他の物体が載っていない。
- place_in(obj, bin_id): 持っている物体を箱に入れる。前提: obj を持っている。
- stack_on(obj, base): 持っている物体を base の上に置く。前提: obj を持っている / base の上が空いている。
- go_home(): 初期姿勢に戻る。計画の最後に必ず入れる。

# ルール
- 物体ID・箱IDはシーン状態に存在するものだけを使う。色・大きさは状態の color / size で判断する。
- pick の直後には、同じ物体の place_in か stack_on を置く（持ったまま別の物体を pick できない）。
- 積み上げは下から順に行う。一番下の物体は動かさなくてよい（テーブルの上にあれば土台になる）。
- 既に目標を満たしている物体は動かさない。
- goal には、指示が完了したときに成り立つべき条件を、関係する全物体について書く。
- 指示の対象が存在しない、または物理的に不可能な場合は feasible=false、plan と goal は空にし、reasoning に理由を書く。
- 曖昧な指示（例:「片付けて」）は、常識的な解釈を1つ選び reasoning に解釈を書く。
- 過去の試行（history）がある場合は、失敗理由と現在の状態を踏まえて、残りの作業だけを計画し直す。
"""


@dataclass
class PlanResult:
    output: PlanOutput | None
    raw_text: str
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    latency_s: float = 0.0
    error: str = ""
    model: str = DEFAULT_MODEL
    retries: int = 0
    extra: dict = field(default_factory=dict)


def _load_dotenv() -> None:
    """プロジェクト直下の .env（git 管理外）から環境変数を読む。値は表示しない。"""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class GeminiPlanner:
    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.0, max_retries: int = 5) -> None:
        import truststore
        from google import genai  # import に時間がかかるので遅延 import

        # OS（Windows）の証明書ストアで TLS 検証する。社内プロキシ・セキュリティソフト環境での
        # "CERTIFICATE_VERIFY_FAILED" を防ぐ
        truststore.inject_into_ssl()
        _load_dotenv()
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            raise RuntimeError("GEMINI_API_KEY が未設定です（.env または環境変数で設定してください）")
        self.client = genai.Client()
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.min_interval_s = float(os.environ.get("GEMINI_MIN_INTERVAL_S", "0"))
        self._last_call = 0.0

    def build_prompt(self, instruction: str, state: dict, history: list[dict]) -> str:
        parts = [f"# 指示\n{instruction}", "# シーン状態\n" + json.dumps(state, ensure_ascii=False)]
        if history:
            parts.append("# これまでの試行（history）\n" + json.dumps(history, ensure_ascii=False, indent=1))
        return "\n\n".join(parts)

    def plan(self, instruction: str, state: dict, history: list[dict] | None = None) -> PlanResult:
        from google.genai import errors, types

        prompt = self.build_prompt(instruction, state, history or [])
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=self.temperature,
            response_mime_type="application/json",
            response_schema=PlanOutput,
        )
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            # 無料枠の「1分あたりリクエスト数」制限に当たらないよう、呼び出し間隔を空ける
            wait = self.min_interval_s - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()
            t0 = time.time()
            try:
                resp = self.client.models.generate_content(model=self.model, contents=prompt, config=config)
            except errors.APIError as e:
                # 429（レート制限）や 5xx は指数バックオフで再試行する
                if getattr(e, "code", None) in (429, 500, 503) and attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                return PlanResult(None, "", error=f"api_error: {e}", model=self.model, retries=attempt)
            latency = time.time() - t0
            u = resp.usage_metadata
            result = PlanResult(
                output=resp.parsed if isinstance(resp.parsed, PlanOutput) else None,
                raw_text=resp.text or "",
                input_tokens=getattr(u, "prompt_token_count", 0) or 0,
                output_tokens=getattr(u, "candidates_token_count", 0) or 0,
                thinking_tokens=getattr(u, "thoughts_token_count", 0) or 0,
                latency_s=latency, model=self.model, retries=attempt,
            )
            if result.output is None:
                result.error = "parse_error: response did not match schema"
            return result
        return PlanResult(None, "", error="api_error: retries exhausted", model=self.model)


def steps_to_calls(plan: PlanOutput) -> list[tuple[str, dict]]:
    """Step（スキーマ）→ Robot.execute に渡す (skill, kwargs) へ変換する。"""
    calls = []
    for s in plan.plan:
        kw = {"pick": {"obj": s.obj}, "place_in": {"obj": s.obj, "bin_id": s.bin_id},
              "stack_on": {"obj": s.obj, "base": s.base}, "go_home": {}}[s.skill.value]
        calls.append((s.skill.value, kw))
    return calls


if __name__ == "__main__":
    # 動作確認: python -m src.planner "赤いブロックを左の箱に入れて"
    import sys

    from src.sim import Sim

    sim = Sim(seed=0)
    planner = GeminiPlanner()
    r = planner.plan(sys.argv[1] if len(sys.argv) > 1 else "赤いブロックを左の箱に入れて", sim.get_state())
    print(r.error or r.output.model_dump_json(indent=1))
    print(f"tokens in={r.input_tokens} out={r.output_tokens} think={r.thinking_tokens} latency={r.latency_s:.2f}s")
