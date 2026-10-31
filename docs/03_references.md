# 参考文献・参考資料

- 関連文書：[01_design.md](01_design.md) / [02_implementation_plan.md](02_implementation_plan.md)
- 凡例：⭐ = 設計で直接参考にしたもの ／ 📖 = 理解を深めるための推奨資料
- 「読む順番のおすすめ」は最後にまとめています。

---

## 1. 講座内の参照（Physical AI 応用編1, 2026）
※講義資料は再配布できないため、タイトルと参考にした考え方のみ記載します。

| 回 | テーマ | 本課題で使った考え方 |
|---|---|---|
| 第1回 | MLOps・RobOps導入 | 要件を数値化した指標（成功率・コスト）で管理する |
| 第2回 | MLOps・RobOps実践 | seed を固定したシミュレータ評価、成功率で判断する、環境の固定 |
| 第3回 | コンピュータアーキテクチャ | 「推測する前に測る」、条件を1つずつ変えて比較する |
| 第4回 | ROS 2入門(1/2) | 非同期分散システムとしてのロボット、パッケージ・環境の再現性 |
| 第5回 | ROS 2入門(2/2)・模倣学習 | データ→学習→デプロイ→評価の流れ、可視化してからデバッグする |
| 第6回 | Open Source in Robotics | シミュレータの役割（学習・評価・安全な試行）、OSS ライセンスの扱い |
| 第7回 | Physical AIの実運用 | 制御の階層構造、層ごとの失敗の切り分け、評価設計（測れていないことの明記） |

⭐ **最終課題サンプル**（課題PDF記載）「ChatGPT API による自然言語ロボット操作」
- https://qiita.com/porizou1/items/a085d20e936946330540
- 本課題のベースライン的な位置づけ。言語→コマンド変換に、成功条件の自己生成・検証・再計画と定量評価を加えて発展させる。

---

## 2. LLM × ロボットの研究（設計思想の背景）

| | 文献 | 本課題との関係 |
|---|---|---|
| ⭐ | Ahn et al., **"Do As I Can, Not As I Say: Grounding Language in Robotic Affordances" (SayCan)**, CoRL 2022. https://arxiv.org/abs/2204.01691 | LLM は「何をするか」を決め、実行可能性はスキル側で担保する、という役割分担 |
| ⭐ | Huang et al., **"Inner Monologue: Embodied Reasoning through Planning with Language Models"**, CoRL 2022. https://arxiv.org/abs/2207.05608 | 実行結果（成功／失敗）を言語で LLM に返して計画を修正する閉ループ |
| ⭐ | Liang et al., **"Code as Policies: Language Model Programs for Embodied Control"**, ICRA 2023. https://arxiv.org/abs/2209.07753 | 定義済みの API（スキル）を LLM に組み合わせさせる構成 |
| 📖 | Singh et al., **"ProgPrompt: Generating Situated Robot Task Plans using Large Language Models"**, ICRA 2023. https://arxiv.org/abs/2209.11302 | プログラム形式のプロンプトと、前提条件（assertion）によるチェック |
| 📖 | Firoozi et al., **"Foundation Models in Robotics: Applications, Challenges, and the Future"**. https://arxiv.org/abs/2312.07843 | ロボット×基盤モデルの全体像（サーベイ） |
| 📖 | Kawaharazuka et al., **"Real-World Robot Applications of Foundation Models: A Review"**, Advanced Robotics. https://arxiv.org/abs/2402.05741 | 第1回ガイダンスで紹介された和書の元になったサーベイ |

---

## 3. シミュレータ・ロボットモデル

| | 資料 | 用途 |
|---|---|---|
| ⭐ | **MuJoCo Documentation**. https://mujoco.readthedocs.io/ | MJCF の書き方、`MjModel`/`MjData`、`mj_jacSite`、レンダリング |
| ⭐ | **MuJoCo Python bindings**. https://mujoco.readthedocs.io/en/stable/python.html | Python API、オフスクリーン描画（`mujoco.Renderer`） |
| ⭐ | **MuJoCo Menagerie**（Google DeepMind）. https://github.com/google-deepmind/mujoco_menagerie | Franka Emika Panda のモデル（`franka_emika_panda/`）。ライセンスは各モデルのフォルダで確認して README に明記する |
| 📖 | Todorov et al., **"MuJoCo: A physics engine for model-based control"**, IROS 2012. | MuJoCo の原論文 |
| 📖 | **MuJoCo チュートリアル（Colab）**. https://github.com/google-deepmind/mujoco#getting-started | 公式ノートブックで基本操作を練習できる |

---

## 4. 逆運動学（IK）

| | 資料 | 用途 |
|---|---|---|
| ⭐ | Samuel R. Buss, **"Introduction to Inverse Kinematics with Jacobian Transpose, Pseudoinverse and Damped Least Squares methods"** (2004). https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf | DLS IK の式と、特異点で減衰が必要な理由 |
| 📖 | Lynch & Park, **"Modern Robotics: Mechanics, Planning, and Control"**, Cambridge Univ. Press, 2017. http://hades.mech.northwestern.edu/index.php/Modern_Robotics | ヤコビアン（5章）、逆運動学（6章）の教科書。無料 PDF・動画あり |
| 📖 | Kevin Zakka, **mink: Python inverse kinematics based on MuJoCo**. https://github.com/kevinzakka/mink | 本格的な IK ライブラリ。自前実装との比較対象 |

---

## 5. LLM（Gemini API）

| | 資料 | 用途 |
|---|---|---|
| ⭐ | **Gemini API — Structured output**. https://ai.google.dev/gemini-api/docs/structured-output | JSON スキーマで出力形式を強制する方法 |
| ⭐ | **Google Gen AI Python SDK (`google-genai`)**. https://github.com/googleapis/python-genai | Python からの呼び出し方 |
| ⭐ | **Gemini API — Pricing / Rate limits**. https://ai.google.dev/pricing ／ https://ai.google.dev/gemini-api/docs/rate-limits | コスト見積もり、無料枠の制限 |
| 📖 | **Gemini Robotics-ER**（課題PDFのサンプル「Gemini Robotics-ER 1.6 を試す」）. https://note.com/npaka/n/n93116b196748 | 画像からの物体認識・空間推論。将来の VLM 化（設計書 11章）の参考 |

---

## 6. 評価・実運用

| | 資料 | 用途 |
|---|---|---|
| 📖 | Chi et al., **"Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"**, RSS 2023. https://arxiv.org/abs/2303.04137 | シミュレーション評価のプロトコル（seed × checkpoint × 初期状態）の例（第7回で紹介） |
| 📖 | TRI LBM Team, **"A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation"**. https://arxiv.org/abs/2507.05331 | 盲検化・初期条件の共有など、ロボット評価設計の注意点（第7回で紹介） |

---

## 7. 読む順番のおすすめ（課題を進めながら）
1. **Step 1〜2 の前**：MuJoCo Python bindings の Getting Started → Modern Robotics 5〜6章（ヤコビアンと IK の直感）→ Buss の DLS の節
2. **Step 3 の前**：SayCan と Inner Monologue の概要（Abstract と図1）→ Gemini の Structured output
3. **Step 4 の前**：第2回・第7回の講義資料（評価設計）を読み直す
4. **レポート執筆時**：Code as Policies・サーベイ論文で、自分の手法の位置づけを確認

> 注：URL は 2026/10/31 時点で筆者が把握しているもの。リンク切れや版の更新があり得るため、
> レポートに載せる前に各リンクを開いて確認してください。
