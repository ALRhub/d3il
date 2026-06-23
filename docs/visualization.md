# 学習後の手法を視覚的に比較する

学習済みの各手法(`agents=...`)を **avoiding** タスクで視覚比較するための仕組みです。
2 種類の出力をサポートします。

| 出力 | 内容 | スクリプト | 依存 |
|---|---|---|---|
| 軌道比較図 (PNG) | 障害物フィールド上に各手法のエンドエフェクタ2D軌道を並べて描画 | `scripts/plot_avoiding_trajectories.py` | numpy, matplotlib |
| ロールアウト動画 (MP4) | MuJoCo オフスクリーン描画した実行動画を手法ごとに記録 → 横並び合成 | `scripts/montage_videos.py` | imageio, opencv |

> 定量比較(成功率・エントロピー)は別途 wandb group で行えます（後述）。

データの生成は学習・評価ループ(`run.py` → `simulation/avoiding_sim.py`)に **オプトインのフック** として組み込まれており、
デフォルトは無効なので通常のベンチマーク実行には影響しません。

---

## 1. クイックスタート

各手法のベンチマークに override を足してデータを出力し、最後に集約スクリプトを実行します。

```bash
# 手法ごとに同じ出力ディレクトリへ（agent_name 別に保存される）
bash scripts/avoiding/ddpm_encdec_benchmark.sh \
    simulation.save_traj_dir=viz/avoiding/traj \
    simulation.record_video_dir=viz/avoiding/video \
    simulation.record_n_videos=3

bash scripts/avoiding/bc_benchmark.sh     simulation.save_traj_dir=viz/avoiding/traj \
    simulation.record_video_dir=viz/avoiding/video simulation.record_n_videos=3
bash scripts/avoiding/beso_benchmark.sh   simulation.save_traj_dir=viz/avoiding/traj \
    simulation.record_video_dir=viz/avoiding/video simulation.record_n_videos=3
# ... 比較したい手法ぶん繰り返す

# 集約して比較成果物を生成
python scripts/plot_avoiding_trajectories.py viz/avoiding/traj -o viz/avoiding/compare.png
python scripts/montage_videos.py            viz/avoiding/video -o viz/avoiding/montage.mp4
```

- 出力ディレクトリは Hydra が実行毎に作業ディレクトリを変更しても **起動時の cwd を基準に解決** されるため、
  複数手法・複数 seed が 1 つのディレクトリに集約されます。
- ファイル名は `<agent_name>_seed<seed>.npz` / `.mp4`。

---

## 2. 設定オプション（`configs/avoiding_config.yaml` の `simulation:` ブロック）

```yaml
simulation:
  _target_: simulation.avoiding_sim.Avoiding_Sim
  ...
  # --- visual comparison hooks (disabled by default) ---
  agent_name: ${agent_name}   # 保存ファイル名に使う手法名
  save_traj_dir: null         # 設定するとエンドエフェクタ軌道を .npz 保存
  record_video_dir: null      # 設定するとロールアウト動画を .mp4 保存
  record_n_videos: 0          # 記録するロールアウト本数（1 run あたり）
  record_backend: egl        # 無視されます: 描画は常に EGL に固定
  record_size: 480            # 描画フレームの一辺(px)
```

| キー | 既定 | 説明 |
|---|---|---|
| `save_traj_dir` | `null` | 軌道 `.npz` の保存先。未指定なら保存しない |
| `record_video_dir` | `null` | 動画 `.mp4` の保存先。未指定なら記録しない |
| `record_n_videos` | `0` | 記録ロールアウト本数。`record_video_dir` 設定時のみ有効 |
| `record_backend` | `egl` | **無視されます**。描画は常に EGL(GPU・ヘッドレス安全)に固定されます |
| `record_size` | `480` | 動画フレームの解像度（正方形） |

> コマンドラインから `simulation.save_traj_dir=...` のように上書きできます（`--multirun` の全 seed に適用）。

---

## 3. 軌道比較図

### データ
評価時、各ロールアウトのエンドエフェクタ位置 `robot.current_c_pos[:2]` の系列を保存します。
`<save_traj_dir>/<agent_name>_seed<seed>.npz` の中身:

| キー | 形状 | 内容 |
|---|---|---|
| `c_pos` | `(N, 251, 2)` | 軌道（末尾はゼロパディング） |
| `lengths` | `(N,)` | 各軌道の実長（パディング前） |
| `modes` | `(N, 9)` | 到達モードの one-hot 的エンコード |
| `successes` | `(N,)` | 成功フラグ (0/1) |
| `agent_name`, `seed` | スカラ | メタ情報 |

`N = n_trajectories`。

### 描画
```bash
python scripts/plot_avoiding_trajectories.py <traj_dir> -o out.png [--max-traj 80] [--cols 4]
```
- npz 内に保存された `agent_name` で手法を**自動グルーピング**し、同名の複数 seed を結合します。
- 手法ごとにサブプロット。障害物(赤円)・フィニッシュライン(緑)・スタート位置(★)を背景に、
  **成功軌道はモード別に色分け**、**失敗軌道は灰色**で重ね描き。
- 各サブプロットのタイトルに成功率と到達モード数を表示します（D3IL のマルチモーダル性の可視化）。

オプション:
- `--max-traj`: 1 手法あたり描画する軌道の上限（既定 80、見やすさのため）
- `--cols`: グリッド列数（0 で自動）

障害物配置は `environments/.../objects/avoiding_objects.py` の定数と一致させています。

---

## 4. ロールアウト動画

### 記録
`record_video_dir` と `record_n_videos>0` を設定すると、評価の最後に単一プロセスで
`record_n_videos` 本のロールアウトを実行し、env のケージカメラ (`bp_cam`) からフレームを取得して
`<record_video_dir>/<agent_name>_seed<seed>.mp4` に書き出します。

- **ヘッドレス描画**: 描画は常に **EGL** に固定されます(`simulation/avoiding_sim.py` がインポート時に
  `MUJOCO_GL=egl` を設定)。EGL は GPU のオフスクリーン OpenGL を使い、ディスプレイ不要で高速です。
  GLFW(ウィンドウ表示=ヘッドレスでクラッシュ)や OSMesa(CPU で低速)へのフォールバックは行いません。
  `record_backend` を指定しても無視され、必ず EGL が使われます(EGL が使えない場合はフォールバックせずエラーになります)。
- 評価本体（マルチプロセス）の成功率計算には影響しません。記録は軽量化のため少数本に留めることを推奨します。

### 横並び合成
```bash
python scripts/montage_videos.py <video_dir> -o out.mp4 [--size 320] [--cols 4] [--fps 30]
```
- 手法ごとに 1 本（最初に見つかった seed）を採用し、ラベル付きのタイルにしてグリッド合成します。
- クリップ長が異なる場合は短い方の**最終フレームを保持**して同期します。

---

## 5. 定量比較（wandb）

各ベンチマークスクリプトは `group=avoiding_<手法>_seeds` を設定し、評価時に以下を `wandb.log` します。
- `Metrics/successes`（成功率）
- `Metrics/entropy`（行動の多様性 / マルチモーダル性）
- `score`（= 成功率 × 0.8 + エントロピー × 0.2）

wandb を有効化（`run_vision.py` は環境変数 `WANDB_PROJECT` 等で制御）すれば、ダッシュボード上で
group 別に手法を比較できます。

---

## 6. 注意点・制約

- **対象は avoiding タスクのみ**です。軌道図はタスク固有（障害物座標前提）。動画フックも `avoiding_sim.py` にのみ実装しています。
  他タスクへ広げる場合は各 `simulation/*_sim.py` に同様の `get_image` ループを追加してください。
- このリポジトリの実行環境はヘッドレスです。`simulation.render` は `False`（=`RenderMode.BLIND`）のままにしてください
  （`True`/`HUMAN` は GLFW ウィンドウを開こうとしてクラッシュします）。動画はそれとは独立したオフスクリーン描画で取得します。
- 描画は EGL(GPU)で行われます。`record_n_videos` を大きくすると評価時間がそのぶん伸びます。
