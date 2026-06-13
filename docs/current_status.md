# Current Project Status

## 当前最高分

当前最高稳定分数是 **0.64**：

- Experiment: `exp_synth_unit_gravity_answer_only_2000`
- Data: official500 + unit1000 + gravity1000
- Format: answer-only
- LoRA: rank4, `in_proj/out_proj`
- max_length: 1024

## 当前最有价值的新方向

当前最值得继续的是：

**Huikang Route B + final-answer anchor**

原因：

- Huikang corpus 有更完整的题型覆盖，尤其是 bit_manipulation、spelling、matching、cipher。
- 纯 Huikang trace 已经证明能带来题型覆盖，但会导致模型输出长 reasoning、不收束。
- 现在需要用 official raw / boxed final-answer anchor 把输出策略拉回来。

## 已验证结论

- `max_length=4096` 不够，8192 对 bit/spelling 有帮助。
- `attn_unembed` 比 full MLP 更稳。
- 不要随便跑 `attn_mlp_unembed`：之前命中 6005 个模块，adapter 过大，风险高。
- 纯 Huikang trace 会导致输出不收束。
- answer-only unit/gravity synthetic 是目前最安全有效的 synthetic 方向。
- 普通 generated CoT、chat template、long CoT 目前都明显负收益。

## 当前正在准备

1. `exp_hk_b1_anchor_weighted_8192_r16`
   - B1 + final-answer anchor + weighted loss。
   - 目标：让 Huikang trace 提供题型覆盖，让 official/synth answer-only 稳定输出。

2. `exp_b1_stage2_official_full_final_r16`
   - 从 B1 adapter 继续训练。
   - 移除 Huikang trace。
   - 使用 full official raw/boxed final-answer + unit/gravity。
   - `lr=5e-5`。
   - 目标：Stage2 输出校正，冲 0.66+。

## 组员需要知道

- 不要随便跑 full MLP / all experts 版本。
- 不要把 adapter、submission.zip、`.safetensors` 上传 GitHub。
- Stage2 实验运行前必须先上传 B1 adapter dataset，并替换占位路径。
- 每次跑实验前先看 [experiments.md](experiments.md)。
- 每个 run 的 `RUN_CONFIG` 是 Kaggle 实际读取的配置，不要只改外部 `config.json`。
