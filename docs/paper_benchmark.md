# `scripts/paper_benchmark.py` — 論文用ベンチマーク実行・集計・作図

一通りのタスク × 手法 × seed を学習・評価し、論文化に必要なデータ（CSV）と図（PDF）を生成するオーケストレーションスクリプトです。各タスクの調整済みベンチスクリプト（`scripts/<task>/<method>_benchmark.sh`）を再利用するため、ベースラインは本来のハイパラのまま実行されます。

---

## 何をするか

1. **学習・評価**: 各 `(task, method, seed)` について `run.py` / `run_vision.py` を1回ずつ起動（`--multirun` のseedスイープは外して1 seedずつ）。指定GPUに固定して実行。
2. **指標収集**: 各 run の評価指標（成功率・entropy・score など）を W&B の run summary から読み取り、共通の**長形式CSV**に追記（`run.py`/`run_vision.py` の `PAPER_CSV` フック経由）。
3. **集計**: seed をまたいだ mean / std を計算し、サマリCSVを出力。
4. **作図**: タスク×手法のグループ棒グラフ（mean ± std）をメトリクスごとに **PDF** で出力。

`--collect-only` を付ければ、既存の結果から集計・作図だけをやり直せます。

---

## 出力物（`--out`、既定 `results/paper/`）

| パス | 内容 |
|---|---|
| `metrics.csv` | 生データ（長形式 / tidy）。1行 = `task, modality, method, seed, window_size, metric, value` |
| `summary_long.csv` | `(modality, task, method, metric)` ごとの `mean, std, n`(seed数) |
| `summary.csv` | ワイド表。1行 = `(modality, task, method)`、列 = `{metric}_mean` / `{metric}_std`（論文の表にそのまま使える） |
| `figures/<modality>_<metric>.pdf` | グループ棒グラフ（x=タスク, 群=手法, 縦=平均, エラーバー=seedのstd） |
| `logs/<task>_<method>_seed<s>.log` | 各 run の標準出力ログ |

収集されるメトリクスは各シミュレータが `wandb.run.summary` に入れる `Metrics/*` と `score`（`Metrics/` 接頭辞は除去）。タスクにより `successes`, `entropy`, `score`, `KL`, `distance`, `entropy_1/2/3` などが入ります。図化される代表メトリクスは `successes`, `entropy`, `score`（無いタスクは自動スキップ）。

---

## クイックスタート

```bash
# 全タスク・全手法・seed 0-5 を GPU2 で実行し、CSV+図を生成
python scripts/paper_benchmark.py --gpu 2

# 既存の結果から サマリCSV と図だけ作り直す（学習はしない）
python scripts/paper_benchmark.py --collect-only

# 何が走るかだけ確認（実行しない）
python scripts/paper_benchmark.py --gpu 2 --dry-run
```

---

## GPU の指定

`--gpu <id>` で実行GPUを固定します（各サブプロセスに `CUDA_VISIBLE_DEVICES=<id>` を設定。torch も jax もこれに従う）。

```bash
python scripts/paper_benchmark.py --gpu 3          # GPU3 を使用
```

- 空いているGPUを指定してください（占有GPUだと特にEBMがOOMします）。空き確認: `nvidia-smi`。
- EBM(jax) の多コア評価（`simulation.n_cores>1`）では、spawnワーカーが可視GPU内で自動的に振り分けられます。1GPUに固定すると全ワーカーがそのGPUに載るため、メモリに注意（`simulation.n_cores` を下げても総試行数=指標は不変）。

---

## EBM 実装（jax / torch）の選択

EBMPolicy には4つの手法名があります。

| method | 実装 | phase1 / phase2 backbone |
|---|---|---|
| `ebm_jax` | JAX | DiT / Transformer-EBM |
| `ebm_jax_mlp` | JAX | MLP / MLP-EBM |
| `ebm_torch` | PyTorch | DiT / Transformer-EBM |
| `ebm_torch_mlp` | PyTorch | MLP / MLP-EBM |

`--ebm-backend` でどの実装を含めるか切り替えます（**非EBMのベースラインは常に実行**）。

```bash
python scripts/paper_benchmark.py --gpu 2 --ebm-backend jax    # ebm_jax / ebm_jax_mlp のみ
python scripts/paper_benchmark.py --gpu 2 --ebm-backend torch  # ebm_torch / ebm_torch_mlp のみ
python scripts/paper_benchmark.py --gpu 2 --ebm-backend both   # 全EBM版（既定）
```

---

## タスク・手法の指定

```bash
# 特定タスクだけ（task キー = scripts/<task> のディレクトリ名）
python scripts/paper_benchmark.py --gpu 2 --tasks avoiding pushing sorting

# 特定手法だけ
python scripts/paper_benchmark.py --gpu 2 --methods ddpm_encdec bc beso ebm_jax ebm_torch

# EBMの2実装だけを比較
python scripts/paper_benchmark.py --gpu 2 --methods ebm_jax ebm_jax_mlp ebm_torch ebm_torch_mlp

# seed を絞る
python scripts/paper_benchmark.py --gpu 2 --seeds 0 1 2
```

- `--tasks all`（既定）のとき、`--modality {state,image,both}` で対象を絞れます。
- `--methods all`（既定）は各タスクの `scripts/<task>/*_benchmark.sh` から手法を自動検出します。

**State タスク**: `avoiding`, `pushing`, `sorting`(=sorting_2), `sorting_4`, `sorting_6`, `stacking`, `aligning`（`run.py`）
**Vision タスク**: `aligning_vision`, `sorting_4_vision`, `sorting_6_vision`, `stacking_vision`（`run_vision.py`）

---

## W&B（指標の記録に必須）

指標は `wandb.run.summary` から取得するため、`--wandb-mode` は `online` か `offline` が必要です（`disabled` だと何も記録されません）。

```bash
# 自分のW&Bへ記録（既定）。事前に `wandb login` 済みであること
python scripts/paper_benchmark.py --gpu 2 --wandb-mode online

# ネットワーク不要（ローカルにsummaryが残る）。CSV収集はこれでも可能
python scripts/paper_benchmark.py --gpu 2 --wandb-mode offline
```

entity / project は環境変数 `WANDB_ENTITY` / `WANDB_PROJECT` でも上書きできます（既定 project は `d3il`、entity 未設定時は自分の既定 entity）。各 run には `task:` / `modality:` / `seed:` と手法名のタグが付きます。

---

## 全オプション

| オプション | 既定 | 説明 |
|---|---|---|
| `--gpu` | `0` | 実行GPU id（`CUDA_VISIBLE_DEVICES`） |
| `--tasks` | `all` | タスクキー列、または `all` |
| `--methods` | `all` | 手法名列、または `all`（スクリプトから自動検出） |
| `--ebm-backend` | `both` | EBM実装の選択 `jax` / `torch` / `both`（非EBMは常時実行） |
| `--seeds` | `0 1 2 3 4 5` | seed列 |
| `--modality` | `both` | `--tasks all` 時の対象 `state` / `image` / `both` |
| `--out` | `results/paper` | CSV・図の出力先 |
| `--wandb-mode` | `online` | `online` / `offline` / `disabled` |
| `--mujoco-gl` | `egl` | `MUJOCO_GL` バックエンド（ヘッドレスは `egl`） |
| `--collect-only` | off | 学習せず、集計CSV+図だけ再生成 |
| `--skip-existing` | off | CSVに既存の `(task,method,seed)` をスキップ（再開に便利） |
| `--dry-run` | off | コマンドを表示するだけで実行しない |

---

## 典型的な進め方

```bash
# 1) まず何が走るか確認
python scripts/paper_benchmark.py --gpu 3 --dry-run

# 2) 実行（途中で止まっても --skip-existing で再開できる）
python scripts/paper_benchmark.py --gpu 3 --skip-existing

# 3) 集計・作図だけやり直したいとき
python scripts/paper_benchmark.py --collect-only
```

複数GPUで並列したい場合は、タスク/手法を分けてGPUごとに別プロセスで起動します（同じ `--out` を共有すればCSVは追記され、最後に `--collect-only` でまとめて集計）。

```bash
python scripts/paper_benchmark.py --gpu 0 --tasks avoiding pushing sorting &
python scripts/paper_benchmark.py --gpu 1 --tasks sorting_4 sorting_6 stacking aligning &
wait
python scripts/paper_benchmark.py --collect-only
```

---

## 注意点

- **GPUメモリ**: 占有GPUを指定するとEBMがOOMします。`nvidia-smi` で空きGPUを選ぶこと。
- **評価の並列数**: 各ベンチスクリプトは `simulation.n_cores`（state は30/60、O(B²)ではない）を指定しています。EBM×多プロセス×1GPUが重い場合は該当スクリプトの `n_cores` を下げても、総試行数（=success/entropy）は不変なので公平性は保たれます。
- **W&B disabled は不可**: 指標が記録されません。`online` か `offline` を使用。
- **`--multirun` を含む特殊なスイープ**: 本スクリプトは `--multirun` を外して seed のみを単一化します。seed以外の値をスイープする稀なスクリプトはそのままでは動かない場合がありますが、主要手法（bc / ddpm_encdec / beso / bet / ebm_* など）は seed のみのスイープなので問題ありません。
- **stacking_vision**: 元のタスク設定に `hydra/launcher: submitit_slurm` が残っており、全手法共通で `hydra-submitit-launcher` が必要です（本スクリプト固有の問題ではありません）。
