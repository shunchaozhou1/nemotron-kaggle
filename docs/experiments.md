# Experiment Log

## Summary Table

| Experiment | Status | Main Idea | Max Length | Rank | Target Modules | Data | LB Score | Conclusion |
| ---------- | ------ | --------- | ---------: | ---: | -------------- | ---- | -------: | ---------- |
| official 500 answer-only baseline | completed | official answer-only | 1024 | 4 | `in_proj/out_proj` | official500 | 0.61-0.62 | 稳定但进入平台期 |
| seed/dropout/epoch/accum sweeps | completed | 参数调优 | 1024 | 4 | `in_proj/out_proj` | official500 | 0.58-0.61 | 未稳定突破 baseline |
| official + generated CoT | completed | 混入 generated CoT | 2048 | 4 | `in_proj/out_proj` | official + CoT | 0.53 | long CoT 明显负收益 |
| official boxed + CoT | completed | boxed 格式 + CoT | 2048 | 4 | `in_proj/out_proj` | official boxed + CoT | 0.59 | boxed 可接受，但 CoT 仍拖累 |
| Unsloth CoT-only 1000 | completed | chat template + CoT-only | 4096 | 8 | Unsloth PEFT config | CoT1000 | 0.40 | 长 CoT / chat template 与评测输出不匹配 |
| CoT dataset answer-only 1000 | completed | 外部 CoT 数据只取 prompt/answer | 1024 | 4 | `in_proj/out_proj` | official500 + CoT answer-only1000 | 0.61 | 稳定但不涨 |
| synthetic v1 answer-only mixed | completed | official-style bit/cipher/roman | 1024 | 4 | `in_proj/out_proj` | official500 + synth1500 | 0.61 | prompt 更像 official 后仍未突破 |
| retrieval top500 official | completed | test-aware official retrieval | 1024 | 4 | `in_proj/out_proj` | official top500 | 0.51 | retrieval 选样分布偏，明显下降 |
| masked bit500 | completed | span-level masked synthetic bit | 1024 | 4 | `in_proj/out_proj` | official500 + masked bit500 | 0.56 | mask 设计未迁移到 LB |
| rank16 / wider target synthetic v1 | completed | 增大 LoRA 容量 | 1024 | 16 | narrow / wide | synthetic v1 | 0.58 | 容量放大噪声 |
| `exp_synth_unit_gravity_answer_only_2000` | completed | solver-backed unit/gravity answer-only | 1024 | 4 | `in_proj/out_proj` | official500 + unit1000 + gravity1000 | **0.64** | 当前最稳 best |
| `exp_synth_unit_gravity_answer_only_3000` | completed | unit/gravity 加量 | 1024 | 4 | `in_proj/out_proj` | official500 + unit1500 + gravity1500 | 0.62 | 单纯加量下降 |
| `exp_solver_multicat_answer_only_v1` | completed | 多类别 solver answer-only | 1024 | 4 | `in_proj/out_proj` | official + 多类别 synthetic | 0.61 | 多类别质量/分布污染 |
| `exp_solver_unit_gravity_masked_1000` | completed | unit/gravity solver trace mask | 1024 | 4 | `in_proj/out_proj` | official500 + masked unit/gravity | 0.61 | masked trace 没有超过 answer-only |
| `exp_mini_corpus_v1_masked_ug` | completed | mini verified masked corpus | 1024 | 4 | `in_proj/out_proj` | official + mini masked UG | 0.63 | 接近 best，但不如 answer-only 2000 |
| `exp_ug_answer1000_mini_masked1000` | completed | answer-only + mini masked 混合 | 1024 | 4 | `in_proj/out_proj` | official + UG answer + mini masked | 0.63 | 混合后未超过 0.64 |
| `exp_huikang_token_segments_subset_v2` | completed | pure Huikang token trace | 4096 | 4 | `in_proj/out_proj` | Huikang token segments | 0.40 | 4096 缺长样本，输出不收束 |
| `exp_hk_format_anchor_v3` | completed | Huikang trace + official anchors | 4096 | 4 | `in_proj/out_proj` | official raw/boxed + safe trace | 0.40 | anchor 被 trace token 淹没 |
| `exp_hk_route_b_full8192_r16_dryrun` | dryrun completed | Route B full MLP target test | 8192 | 16 | `attn_mlp_unembed` | Huikang trace | TBD | 命中 6005 modules，adapter 2.4G，太激进 |
| `exp_hk_route_b_8192_r16_attn_unembed` | completed | Route B attn + lm_head | 8192 | 16 | `attn_unembed` | Huikang trace | 0.59 | 覆盖 bit/spelling 后提升，但纯 trace 仍不稳 |
| `exp_hk_b1_anchor_weighted_8192_r16` | pending / dryrun completed | B1 + final-answer anchors + weighted loss | 8192 | 16 | `attn_unembed` | Huikang trace + answer anchors | TBD | 正式版待跑或待评估 |
| `exp_b1_stage2_official_full_final_r16` | prepared | B1 adapter Stage2 final-answer 续训 | 4096 | 16 | B1 adapter config | full official raw/boxed + UG | TBD | 目标是输出策略校正 |

## Detailed Records

### Experiment: official 500 answer-only baseline

- Status: completed
- Date: 2026-05
- Goal: 建立最小稳定 LoRA baseline。
- Data: official train.csv 随机 500，answer-only。
- Config:
  - max_length: 1024
  - rank: 4
  - target_modules: `.*\.(in_proj|out_proj)$`
  - lr: 2e-4
  - batch_size: 1
  - grad_accum: 4
  - epochs: 1
- Key Logs:
  - trainable params: TBD
  - adapter size: TBD
  - debug generation: TBD
- LB Score: 0.61-0.62
- Result Analysis: 稳定，参数调优后仍难突破。
- Conclusion: 作为后续实验比较基线。
- Next Action: 不再盲目调 seed/dropout/epoch。

### Experiment: parameter sweep runs

- Status: completed
- Date: 2026-05
- Goal: 测试 seed、dropout、epoch、grad accumulation 是否能突破 baseline。
- Data: official500 answer-only。
- Config:
  - max_length: 1024
  - rank: 4
  - target_modules: `in_proj/out_proj`
  - lr: 2e-4
  - batch_size: 1
  - grad_accum: 2 or 4
  - epochs: 1 or 2
- Key Logs:
  - trainable params: TBD
  - adapter size: TBD
  - debug generation: TBD
- LB Score:
  - `exp_seed7`: 0.60
  - `exp_seed99`: 0.59
  - `exp_seed2025`: 0.60
  - `exp_dropout0`: 0.61
  - `exp_epoch2`: 0.58
  - `exp_accum2`: 0.60
- Result Analysis: 变动没有稳定收益，epoch2 可能过拟合或偏移。
- Conclusion: 轻量 official answer-only 上限约 0.62。
- Next Action: 转向数据质量和题型覆盖。

### Experiment: CoT / generated reasoning attempts

- Status: completed
- Date: TBD
- Goal: 验证 generated CoT、boxed CoT、chat template 是否提升 reasoning。
- Data: official + external CoT 或 CoT-only。
- Config:
  - max_length: 2048 / 4096
  - rank: 4 / 8
  - target_modules: varies
  - lr: 2e-4
  - batch_size: 1
  - grad_accum: 4 / 16
  - epochs: 1
- Key Logs:
  - trainable params: TBD
  - adapter size: TBD
  - debug generation: long reasoning / unstable final answer
- LB Score:
  - official + generated CoT: 0.53
  - official boxed + CoT: 0.59
  - Unsloth CoT-only 1000: 0.40
  - CoT answer-only 1000: 0.61
- Result Analysis: 普通 generated CoT 和 chat template 与 leaderboard 输出格式不匹配，长推理模式伤害 final answer。
- Conclusion: 不继续普通 CoT 路线。
- Next Action: 只保留 solver-backed 或 verified corpus 的可控监督。

### Experiment: synthetic v1 answer-only family

- Status: completed
- Date: TBD
- Goal: official-style synthetic bit/text_cipher/roman 是否提升。
- Data: official500 + synthetic v1。
- Config:
  - max_length: 1024
  - rank: 4
  - target_modules: `in_proj/out_proj`
  - lr: 2e-4
  - batch_size: 1
  - grad_accum: 4
  - epochs: 1
- Key Logs:
  - trainable params: TBD
  - adapter size: TBD
  - debug generation: TBD
- LB Score:
  - mixed1500: 0.61
  - single type bit/cipher/roman: around 0.61
- Result Analysis: prompt 风格更像 official 后仍不涨，可能题型覆盖或 solver 难度不足。
- Conclusion: 低质量 synthetic 加量没有价值。
- Next Action: 优先 solver-backed unit/gravity。

### Experiment: `exp_synth_unit_gravity_answer_only_2000`

- Status: completed
- Date: TBD
- Goal: 用已经验证有效的 unit_conversion 和 gravity solver-backed answer-only 数据提升 baseline。
- Data: official500 + unit1000 + gravity1000。
- Config:
  - max_length: 1024
  - rank: 4
  - target_modules: `in_proj/out_proj`
  - lr: 2e-4
  - batch_size: 1
  - grad_accum: 4
  - epochs: 1
- Key Logs:
  - trainable params: TBD
  - adapter size: TBD
  - debug generation: TBD
- LB Score: 0.64
- Result Analysis: 当前唯一明确提升方向，说明 deterministic solver answer-only 数据有效。
- Conclusion: 当前 best。
- Next Action: 不盲目加量，寻找更高质量题型或结合 Huikang 覆盖。

### Experiment: `exp_solver_multicat_answer_only_v1`

- Status: completed
- Date: TBD
- Goal: 扩展 unit/gravity 到 numeral、splitting、concatenation、lstrip 等多类别 solver 数据。
- Data: official500 + unit800 + gravity800 + numeral500 + splitting500 + concatenation500 + lstrip300。
- Config:
  - max_length: 1024
  - rank: 4
  - target_modules: `in_proj/out_proj`
  - lr: 2e-4
  - batch_size: 1
  - grad_accum: 4
  - epochs: 1
- Key Logs:
  - trainable params: TBD
  - adapter size: TBD
  - debug generation: TBD
- LB Score: 0.61
- Result Analysis: 多类别 synthetic 可能带来分布污染，尤其 equation / string 类规则与 test 分布不完全一致。
- Conclusion: 多类别不是越多越好。
- Next Action: 回到 validated categories 或 Huikang corpus。

### Experiment: Huikang token trace v2 / v3

- Status: completed
- Date: TBD
- Goal: 利用 Huikang-style token-level masked corpus。
- Data:
  - `exp_huikang_token_segments_subset_v2`: pure Huikang token segment。
  - `exp_hk_format_anchor_v3`: official raw500 + official boxed500 + Huikang safe trace1500。
- Config:
  - max_length: 4096
  - rank: 4
  - target_modules: `in_proj/out_proj`
  - lr: 2e-4
  - batch_size: 1
  - grad_accum: 4
  - epochs: 1
- Key Logs:
  - trainable params: TBD
  - adapter size: TBD
  - debug generation: boxed 格式可学，但 final answer 不稳
- LB Score: 0.40
- Result Analysis: 4096 会丢长样本；trace token 主导，模型输出长推理/表格，和提交格式不匹配。
- Conclusion: Huikang 数据不能低配直接用。
- Next Action: max_length 8192 + 更合理 target scope。

### Experiment: `exp_hk_route_b_full8192_r16_dryrun`

- Status: dryrun completed
- Date: TBD
- Goal: 尽量复刻 Huikang Route B，使用 8192、rank16、attn+MLP+unembed。
- Data: Huikang high-confidence categories。
- Config:
  - max_length: 8192
  - rank: 16
  - target_modules: `attn_mlp_unembed`
  - lr: 2e-4
  - batch_size: 1
  - grad_accum: 32
  - epochs: 1
- Key Logs:
  - trainable params: 444,077,056
  - adapter size: 2.4G
  - submission.zip: 1.7G
  - matched target module count: 6005
- LB Score: dryrun only
- Result Analysis: 命中所有 MoE experts 的 up/down projections，过宽过重。
- Conclusion: 不跑正式版。
- Next Action: 改为 `attn_unembed`。

### Experiment: `exp_hk_route_b_8192_r16_attn_unembed`

- Status: completed
- Date: TBD
- Goal: 用 8192 覆盖 bit/spelling，同时只训练 attention + lm_head。
- Data: Huikang selected categories。
- Config:
  - max_length: 8192
  - rank: 16
  - target_modules: `in_proj/out_proj/q_proj/k_proj/v_proj/o_proj/lm_head`
  - lr: 2e-4
  - batch_size: 1
  - grad_accum: 32
  - epochs: 1
- Key Logs:
  - trainable params: TBD
  - adapter size: TBD
  - debug generation: 仍有长 reasoning / 不收束
- LB Score: 0.59
- Result Analysis: 8192 和类别覆盖带来明显提升，但 pure trace 仍伤 final answer 输出。
- Conclusion: Huikang 数据有价值，但需要 final-answer anchor 或 Stage2。
- Next Action: weighted anchor / Stage2。

### Experiment: `exp_hk_b1_anchor_weighted_8192_r16`

- Status: dryrun completed; full pending / TBD
- Date: TBD
- Goal: 在 B1 上加入 official/synthetic final-answer anchor，并降低 Huikang trace loss 权重。
- Data: Huikang trace + official raw/boxed + unit/gravity answer-only。
- Config:
  - max_length: 8192
  - rank: 16
  - target_modules: `attn_unembed`
  - lr: 2e-4
  - batch_size: 1
  - grad_accum: 32
  - epochs: 1
- Key Logs:
  - trainable params: TBD
  - adapter size: TBD
  - debug generation: TBD
- LB Score: TBD
- Result Analysis: TBD
- Conclusion: 目标是平衡题型覆盖和 final-answer 输出。
- Next Action: 跑正式版并评估是否超过 0.59 / 0.64。

### Experiment: `exp_b1_stage2_official_full_final_r16`

- Status: prepared
- Date: TBD
- Goal: 从 B1 adapter 继续训练，只用 final-answer 数据校正输出策略。
- Data: official raw9500 + official boxed9500 + unit1000 + gravity1000。
- Config:
  - max_length: 4096
  - rank: 16
  - target_modules: B1 adapter config / `attn_unembed`
  - lr: 5e-5
  - batch_size: 1
  - grad_accum: 32
  - epochs: 1
- Key Logs:
  - trainable params: TBD
  - adapter size: TBD
  - debug generation: TBD
- LB Score: TBD
- Result Analysis: TBD
- Conclusion: 目标是保留 B1 题型覆盖，同时拉回短答案 / boxed final answer。
- Next Action: 上传 B1 adapter dataset，替换 `PLEASE_REPLACE_WITH_B1_ADAPTER_DATASET`，先跑 dryrun。
