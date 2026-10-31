# 外部アセットの出典

| フォルダ | 出典 | バージョン | ライセンス |
|---|---|---|---|
| `franka_emika_panda/` | [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) | commit `a662de603224c47123b0783821c58ad4402ad3e3` | Apache-2.0（`franka_emika_panda/LICENSE`） |

取得方法（再現用）:
```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git
cd mujoco_menagerie && git sparse-checkout set franka_emika_panda
```
