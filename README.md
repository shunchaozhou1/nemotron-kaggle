# NVIDIA Nemotron Reasoning Challenge Experiments

## 1. Project Overview

本项目用于参加 Kaggle **NVIDIA Nemotron Model Reasoning Challenge**。

基础模型是 **Nemotron-3-Nano-30B-A3B-BF16**。我们的目标是通过 PEFT LoRA / SFT 训练 adapter，让模型在规则推理类任务上获得更高 LB 分数，重点题型包括 bit manipulation、unit conversion、gravity、numeral、cipher、matching、splitting、concatenation、spelling 等。

主要运行环境是 Kaggle GPU，默认禁网。训练依赖 Kaggle competition input、attached datasets、`ryanholbrook/nvidia-utility-script`，并在 `/kaggle/working/adapter` 保存 adapter，打包 `/kaggle/working/submission.zip` 提交。

当前仓库的核心目标是：

- 管理不同 Kaggle run 的独立实验目录。
- 保留可复现的 `sft_train.py` 与 `kernel-metadata.json`。
- 记录实验配置、LB 结果、失败原因和下一步计划。
- 避免把大型 adapter / submission 文件提交到 Git。

## 2. Repository Structure

当前主要结构：

```text
.
├── README.md
├── .gitignore
├── experiments.md
├── kaggle_kernel/
│   ├── kernel-metadata.json
│   └── sft_train.py
├── runs/
│   ├── exp_synth_unit_gravity_answer_only_2000/
│   │   ├── sft_train.py
│   │   └── kernel-metadata.json
│   ├── exp_hk_route_b_8192_r16_attn_unembed/
│   │   ├── sft_train.py
│   │   └── kernel-metadata.json
│   ├── exp_hk_b1_anchor_weighted_8192_r16/
│   │   ├── sft_train.py
│   │   └── kernel-metadata.json
│   └── ...
├── scripts/
│   ├── build_mini_corpus.py
│   ├── convert_huikang_subset.py
│   ├── generate_synthetic_core.py
│   ├── inspect_huikang_corpus.py
│   ├── inspect_huikang_loadable.py
│   ├── inspect_mini_corpus.py
│   ├── inspect_official_prompts.py
│   └── validate_mini_corpus.py
├── research/
│   ├── high_score_methods.md
│   └── high_score_methods_v2.md
└── docs/
    ├── current_status.md
    └── experiments.md
```

`runs/` 下每个实验目录都是一个可独立 `kaggle kernels push` 的 Kaggle script kernel。

## 3. Current Best Results

| Experiment | Main Idea | Max Length | Rank | Target Modules | Data | LB Score | Notes |
| ---------- | --------- | ---------: | ---: | -------------- | ---- | -------: | ----- |
| official 500 answer-only baseline | official answer-only LoRA | 1024 | 4 | `in_proj/out_proj` | official500 | 0.61-0.62 | 早期最稳 baseline |
| `exp_synth_unit_gravity_answer_only_2000` | solver-backed answer-only | 1024 | 4 | `in_proj/out_proj` | official500 + unit1000 + gravity1000 | **0.64** | 当前最稳 answer-only baseline |
| `exp_solver_multicat_answer_only_v1` | 多类别 solver synthetic answer-only | 1024 | 4 | `in_proj/out_proj` | official500 + unit/gravity/numeral/splitting/concat/lstrip | 0.61 | 多类别 synthetic 不如 unit+gravity 稳，可能有分布污染 |
| `exp_huikang_token_segments_subset_v2` | pure Huikang token trace | 4096 | 4 | `in_proj/out_proj` | Huikang token segments | 0.40 | 4096 缺 bit/spelling，trace 主导导致输出不稳定 |
| `exp_hk_format_anchor_v3` | trace + official format anchors | 4096 | 4 | `in_proj/out_proj` | official raw500 + boxed500 + Huikang safe trace1500 | 0.40 | boxed 格式学会，但 reasoning 能力没有有效迁移；trace token 淹没 anchor |
| `exp_hk_route_b_8192_r16_attn_unembed` | Huikang Route B | 8192 | 16 | `in_proj/out_proj/q_proj/k_proj/v_proj/o_proj/lm_head` | Huikang selected categories | 0.59 | 8192 覆盖 bit/spelling，有明显提升；纯 trace 仍导致长推理不收束 |
| `exp_hk_b1_anchor_weighted_8192_r16` | B1 + final-answer anchor + weighted loss | 8192 | 16 | `attn_unembed` | Huikang trace + official/synth answer anchors | TBD | dryrun completed，正式版待跑或待评估 |
| `exp_b1_stage2_official_full_final_r16` | B1 adapter Stage2 final-answer 续训 | 4096 | 16 | B1 adapter config / `attn_unembed` | full official raw/boxed + unit/gravity | TBD | 准备中；目标是把 B1 输出策略拉回 final answer |

更多实验和失败分析见 [docs/experiments.md](docs/experiments.md)。

## 4. Main Findings

- answer-only 路线最稳定。目前 best 是 `exp_synth_unit_gravity_answer_only_2000`，LB=0.64，来自 official500 + unit1000 + gravity1000。
- Huikang trace 数据有价值，但不能直接主导训练。纯 trace 容易让模型进入长 reasoning、表格、反复枚举模式，最终答案输出不稳定。
- `max_length=4096` 对 Huikang corpus 不够，容易丢失 bit_manipulation 和 spelling 等长样本；提升到 8192 后，B1 从 0.40 提升到 0.59。
- `target_modules` 不能盲目扩到所有 MLP experts。`attn_mlp_unembed` 曾命中 6005 个模块，adapter 达 2.4G，风险很高。
- 当前较稳的 target scope 是 `attn_unembed`，只训练 attention projection 和 `lm_head`。
- 后续高分方向应该是：Huikang 题型覆盖 + final-answer anchor + 合理 loss weight，或者先训练 B1 adapter，再接 Stage2 final-answer 续训。

## 5. Experiment Design Philosophy

我们现在的实验原则：

- 不盲目跑大实验，先 dryrun 验证：
  - target modules 是否命中正确。
  - 是否 OOM。
  - adapter size 是否合理。
  - debug generation 是否能输出 final answer。
- 每个 run 独立目录，至少包含：
  - `sft_train.py`
  - `kernel-metadata.json`
- 每次实验记录：
  - `data_mode`
  - dataset composition
  - `max_length`
  - LoRA rank / alpha / dropout
  - target modules / target scope
  - trainable params
  - adapter size
  - debug generation result
  - LB score
  - conclusion

## 6. How to Run

在本地进入某个实验目录：

```powershell
cd runs/<experiment_name>


kaggle kernels push -p . --accelerator NvidiaRtxPro6000
```

注意：

- 不要提交 Kaggle API token。
- 不要把大型 adapter、`submission.zip`、`.safetensors` 文件放进 Git。
- Kaggle dataset 路径和本地磁盘路径不同，运行前检查 `RUN_CONFIG` 中的 `/kaggle/input/...`。
- Stage2 续训实验需要先上传 B1 adapter dataset，并替换 `PLEASE_REPLACE_WITH_B1_ADAPTER_DATASET`。


