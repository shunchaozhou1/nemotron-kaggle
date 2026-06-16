# Experiment Log

This file records the main experiment trajectory for the NVIDIA Nemotron Model Reasoning Challenge.

The overall progress can be divided into four stages:

1. **Stage 1: Answer-only baseline on official data**
2. **Stage 2: Synthetic data augmentation**
3. **Stage 3: Conventional CoT and trace-style exploration**
4. **Stage 4: Clean CoT reproduction and high-score experiment**

---

## Current Best Result

| Best Experiment                           | Main Idea                                                   | Public LB | Notes                                             |
| ----------------------------------------- | ----------------------------------------------------------- | --------: | ------------------------------------------------- |
| `exp_reproduce_085_clean_cot_r32_8192`    | Clean CoT + boxed final answer + rank32 LoRA + 8192 context |  **0.85** | Current best result                               |
| `exp_synth_unit_gravity_answer_only_2000` | official500 + unit1000 + gravity1000, answer-only           |      0.64 | Best lightweight answer-only baseline             |
| `exp_hk_route_b_8192_r16_attn_unembed`    | Huikang trace, 8192 length, rank16, attention + lm_head     |      0.59 | Shows long trace has value but output is unstable |

---

# Stage 1: Answer-only Baseline on Official Data

## Motivation

The first goal was to verify the full training and submission pipeline:

```text
official data
→ prompt / answer construction
→ LoRA SFT
→ adapter saving
→ submission.zip
→ Kaggle evaluation
```

At this stage, we used only official training data and focused on a stable answer-only objective.

The core construction is:

```text
input_ids = prompt_ids + answer_ids
labels = [-100] * len(prompt_ids) + answer_ids
```

That means the prompt is used as context, but only the answer tokens contribute to the loss.

---

## Stage 1 Experiments

| Experiment                         | Data        | Format      | Max Length | Rank | Target Modules     |        LB | Conclusion                                             |
| ---------------------------------- | ----------- | ----------- | ---------: | ---: | ------------------ | --------: | ------------------------------------------------------ |
| `official500_answer_only_baseline` | official500 | answer-only |       1024 |    4 | `in_proj/out_proj` | 0.61–0.62 | Stable baseline; pipeline works                        |
| `exp_seed7`                        | official500 | answer-only |       1024 |    4 | `in_proj/out_proj` |      0.60 | Worse than baseline                                    |
| `exp_seed99`                       | official500 | answer-only |       1024 |    4 | `in_proj/out_proj` |      0.59 | Worse than baseline                                    |
| `exp_seed2025`                     | official500 | answer-only |       1024 |    4 | `in_proj/out_proj` |      0.60 | Worse than baseline                                    |
| `exp_dropout0`                     | official500 | answer-only |       1024 |    4 | `in_proj/out_proj` |      0.61 | Slightly worse than baseline                           |
| `exp_epoch2`                       | official500 | answer-only |       1024 |    4 | `in_proj/out_proj` |      0.58 | More training caused overfitting or distribution shift |
| `exp_accum2`                       | official500 | answer-only |       1024 |    4 | `in_proj/out_proj` |      0.60 | More optimizer updates did not help                    |
| `nemotron-sft-exp002-v3`           | official500 | answer-only |       1024 |    4 | `in_proj/out_proj` |      0.59 | CLI reproduced SFT run, but did not exceed baseline    |

## Stage 1 Summary

The answer-only baseline reached a stable plateau around **0.61–0.62**.
Changing seed, dropout, epoch number, or gradient accumulation did not provide stable improvement.

Main conclusion:

```text
The bottleneck is not ordinary hyperparameter tuning.
The next improvement should come from data construction and training objective design.
```

---

# Stage 2: Synthetic Data Augmentation

## Motivation

After the official500 baseline reached a plateau, we introduced synthetic data to increase task coverage.

The key question was:

```text
Can self-generated rule-based data improve generalization?
```

Synthetic data is useful only when:

1. the rule is clear;
2. the answer is reliable;
3. the prompt style is close to the competition;
4. the synthetic data ratio is controlled.

---

## Stage 2 Experiments

| Experiment                                | Main Idea                                 | Data                                 | Format          | Max Length | Rank | Target Modules     |       LB | Conclusion                                                 |
| ----------------------------------------- | ----------------------------------------- | ------------------------------------ | --------------- | ---------: | ---: | ------------------ | -------: | ---------------------------------------------------------- |
| `synthetic_v1_answer_only_mixed`          | official-style synthetic bit/cipher/roman | official500 + synth1500              | answer-only     |       1024 |    4 | `in_proj/out_proj` |     0.61 | Synthetic style improved but did not exceed baseline       |
| `exp_official_retrieval_500`              | test-aware official retrieval             | official top500                      | answer-only     |       1024 |    4 | `in_proj/out_proj` |     0.51 | Retrieval distribution was biased                          |
| `exp_synth_v2_masked_bit500`              | masked bit synthetic                      | official500 + masked bit500          | masked / answer |       1024 |    4 | `in_proj/out_proj` |     0.56 | Mask design did not transfer well                          |
| `exp_synth_v2_masked_bit1000`             | more masked bit data                      | official500 + masked bit1000         | masked / answer |       1024 |    4 | `in_proj/out_proj` |      TBD | No stable improvement recorded                             |
| `exp_synth_core_v1_rank16`                | larger LoRA rank                          | synthetic core                       | answer-only     |       1024 |   16 | narrow target      |     0.58 | Increasing capacity amplified noise                        |
| `exp_synth_core_v1_rank16_wide`           | wider target modules                      | synthetic core                       | answer-only     |       1024 |   16 | wider target       |     0.58 | Wider LoRA did not help noisy data                         |
| `exp_synth_unit_answer_only_1000`         | unit conversion synthetic                 | official500 + unit1000               | answer-only     |       1024 |    4 | `in_proj/out_proj` |      TBD | Useful direction                                           |
| `exp_synth_gravity_answer_only_1000`      | gravity synthetic                         | official500 + gravity1000            | answer-only     |       1024 |    4 | `in_proj/out_proj` |      TBD | Useful direction                                           |
| `exp_synth_unit_gravity_answer_only_1000` | unit + gravity small mix                  | official500 + unit/gravity           | answer-only     |       1024 |    4 | `in_proj/out_proj` |      TBD | Better than generic synthetic                              |
| `exp_synth_unit_gravity_answer_only_2000` | solver-backed unit/gravity                | official500 + unit1000 + gravity1000 | answer-only     |       1024 |    4 | `in_proj/out_proj` | **0.64** | Best lightweight answer-only result                        |
| `exp_synth_unit_gravity_answer_only_3000` | more unit/gravity data                    | official500 + unit1500 + gravity1500 | answer-only     |       1024 |    4 | `in_proj/out_proj` |     0.62 | Too much synthetic data hurt generalization                |
| `exp_solver_multicat_answer_only_v1`      | multi-category solver synthetic           | official + multi-category synthetic  | answer-only     |       1024 |    4 | `in_proj/out_proj` |     0.61 | More categories introduced noise or distribution pollution |
| `exp_solver_unit_gravity_masked_1000`     | unit/gravity masked trace                 | official500 + masked unit/gravity    | masked trace    |       1024 |    4 | `in_proj/out_proj` |     0.61 | Masked trace did not exceed answer-only                    |
| `exp_mini_corpus_v1_masked_ug`            | mini verified masked corpus               | official + mini masked UG            | masked          |       1024 |    4 | `in_proj/out_proj` |     0.63 | Close to best, but still below 0.64                        |
| `exp_ug_answer1000_mini_masked1000`       | answer-only + mini masked mix             | official + UG answer + mini masked   | mixed           |       1024 |    4 | `in_proj/out_proj` |     0.63 | Mixed objective did not exceed pure answer-only UG         |

## Stage 2 Summary

The most important result in this stage was:

```text
exp_synth_unit_gravity_answer_only_2000
Public LB = 0.64
```

This showed that self-generated data can help, but only when it is clean, rule-based, and answer-reliable.

Main conclusion:

```text
Synthetic data is not automatically useful.
Solver-backed unit/gravity data works because the answers are reliable and the rules are clear.
Generic multi-category synthetic data may introduce distribution noise.
```

---

# Stage 3: Conventional CoT and Trace-style Exploration

## Motivation

Since the task requires rule induction, we tried adding reasoning processes.
The initial assumption was:

```text
If the model learns intermediate reasoning steps, it may solve hidden-rule problems better.
```

However, the experiments showed that unconstrained long reasoning often hurts final-answer accuracy.

---

## Stage 3A: Conventional CoT Experiments

| Experiment                        | Main Idea                                    | Data                               | Format              | Max Length | Rank | Target Modules      |   LB | Conclusion                                        |
| --------------------------------- | -------------------------------------------- | ---------------------------------- | ------------------- | ---------: | ---: | ------------------- | ---: | ------------------------------------------------- |
| `official_generated_cot`          | mix generated CoT with official data         | official + generated CoT           | CoT                 |       2048 |    4 | `in_proj/out_proj`  | 0.53 | Long CoT clearly hurt performance                 |
| `official_boxed_cot`              | add boxed answer format to CoT               | official boxed + CoT               | boxed CoT           |       2048 |    4 | `in_proj/out_proj`  | 0.59 | Boxed format helped, but CoT still hurt           |
| `exp_unsloth_cot_1000_r8_len4096` | Unsloth CoT-only                             | CoT1000                            | chat template / CoT |       4096 |    8 | Unsloth PEFT config | 0.40 | Long CoT and chat format did not match evaluation |
| `exp_cotdata_answer_only_1000`    | external CoT data but only answer is trained | official500 + CoT answer-only1000  | answer-only         |       1024 |    4 | `in_proj/out_proj`  | 0.61 | Stable but no improvement                         |
| `exp_cotdata_answer_only_core500` | smaller CoT answer-only subset               | official500 + core CoT answer-only | answer-only         |       1024 |    4 | `in_proj/out_proj`  |  TBD | No clear improvement recorded                     |

## Stage 3A Summary

Conventional CoT failed because it changed the output behavior of the model.

Observed problems:

1. model tends to output long explanations;
2. final answer appears too late;
3. output format becomes unstable;
4. reasoning text may not match the final answer;
5. evaluation only cares about the final answer.

Main conclusion:

```text
CoT itself is not necessarily bad.
The problem is unconstrained long reasoning.
For this competition, final-answer stability is essential.
```

---

## Stage 3B: Huikang Trace and Route B Exploration

## Motivation

After conventional CoT failed, we explored more structured trace-style data from the Huikang pipeline.
The goal was to use richer process supervision and broader task coverage, especially for longer and harder categories such as bit manipulation, spelling, matching, and trace-heavy rules.

---

## Huikang Trace Experiments

| Experiment                              | Main Idea                                      | Data                                            |  Max Length | Rank | Target Modules     |     LB | Conclusion                                                        |
| --------------------------------------- | ---------------------------------------------- | ----------------------------------------------- | ----------: | ---: | ------------------ | -----: | ----------------------------------------------------------------- |
| `exp_huikang_token_segments_subset_v1`  | early Huikang token segment loading            | Huikang token segments                          |        4096 |    4 | `in_proj/out_proj` |    TBD | Loader / format exploration                                       |
| `exp_huikang_token_segments_subset_v2`  | pure Huikang token trace                       | Huikang token segments                          |        4096 |    4 | `in_proj/out_proj` |   0.40 | 4096 missed long samples; output did not converge to final answer |
| `exp_hk_format_anchor_v3`               | Huikang trace + official raw/boxed anchors     | official raw/boxed + safe trace                 |        4096 |    4 | `in_proj/out_proj` |   0.40 | Anchors were overwhelmed by long trace tokens                     |
| `exp_hk_route_b_full8192_r16_dryrun`    | full Route B MLP target test                   | Huikang trace                                   |        8192 |   16 | `attn_mlp_unembed` | dryrun | Matched 6005 modules; adapter too large and too aggressive        |
| `exp_hk_route_b_8192_r16_attn_unembed`  | Route B with 8192 context                      | Huikang trace                                   |        8192 |   16 | `attn_unembed`     |   0.59 | Long context helped, but pure trace still caused unstable output  |
| `exp_hk_b1_anchor_weighted_8192_r16`    | Route B + final-answer anchors + weighted loss | Huikang trace + official anchors + unit/gravity |        8192 |   16 | `attn_unembed`     |    TBD | Dryrun completed; formal LB pending                               |
| `exp_b1_stage2_official_full_final_r16` | planned Stage2 final-answer correction         | B1 adapter + final-answer data                  | 4096 / 8192 |   16 | `attn_unembed`     |    TBD | Planned; requires B1 adapter dataset path                         |

## Stage 3B Summary

Huikang trace provided useful coverage, but raw trace training was not enough.

Important observations:

1. `4096` context length was insufficient for long trace categories.
2. Moving to `8192` improved performance from `0.40` to `0.59`.
3. Pure trace training made the model produce long reasoning instead of stable final answers.
4. Official boxed anchors were too weak if trace loss dominated.
5. Weighted loss or Stage2 final-answer correction became necessary.

Main conclusion:

```text
Trace data has useful reasoning coverage, but it must be controlled.
Raw long trace alone does not guarantee high final-answer accuracy.
```

---

# Stage 4: Clean CoT Reproduction and High-score Experiment

## Motivation

The previous stages led to a clearer understanding:

1. answer-only is stable but limited;
2. synthetic data helps only when clean and reliable;
3. unconstrained CoT hurts final-answer output;
4. raw long trace provides coverage but causes output instability.

Therefore, the final high-score direction was:

```text
Use reasoning supervision,
but make it clean, self-consistent, and tied to boxed final answer.
```

---

## Clean CoT Strong Reproduction

| Experiment                             | Main Idea                                    | Data                           | Format                                               | Max Length | Rank | Target Modules                | Adapter Size |  Zip Size |       LB | Conclusion   |
| -------------------------------------- | -------------------------------------------- | ------------------------------ | ---------------------------------------------------- | ---------: | ---: | ----------------------------- | -----------: | --------: | -------: | ------------ |
| `exp_reproduce_085_clean_cot_r32_8192` | reproduce public clean CoT high-score method | clean CoT + augmented examples | chat template + clean reasoning + boxed final answer |       8192 |   32 | `q/k/v/o/in/out/up/down_proj` |    3373.4 MB | 3094.7 MB | **0.85** | Current best |

---

## Core Design

The sample format is conceptually:

```text
User:
[prompt]
Please put the final answer in \boxed{}.

Assistant:
[clean reasoning]
</think>
\boxed{answer}
```

Compared with conventional CoT, this version is different because:

1. the reasoning is cleaned;
2. the final answer is explicitly boxed;
3. the model sees a consistent final-answer format;
4. the training uses long context;
5. the LoRA capacity is much larger.

---

## Key Configuration

| Parameter             | Value                                                            |
| --------------------- | ---------------------------------------------------------------- |
| Base model            | `Nemotron-3-Nano-30B-A3B-BF16`                                   |
| Max length            | 8192                                                             |
| LoRA rank             | 32                                                               |
| LoRA alpha            | 32                                                               |
| LoRA dropout          | 0.0                                                              |
| Target modules        | `q_proj/k_proj/v_proj/o_proj/in_proj/out_proj/up_proj/down_proj` |
| Epochs                | 1                                                                |
| Learning rate         | 2e-4                                                             |
| Scheduler             | cosine                                                           |
| Warmup ratio          | 0.03                                                             |
| Batch size            | 2                                                                |
| Gradient accumulation | 16                                                               |
| Effective batch size  | 32                                                               |
| Precision             | bf16                                                             |
| Sampling              | type-stratified batching                                         |

---

## Stage 4 Summary

This experiment achieved:

```text
Public LB = 0.85
```

Main conclusion:

```text
The issue was not that reasoning is useless.
The issue was that ordinary long reasoning is unstable.

Clean CoT works because it combines:
1. reasoning process supervision;
2. long context;
3. larger LoRA capacity;
4. category-balanced training;
5. stable boxed final-answer output.
```

---

# Overall Summary

## Four-stage Progress

| Stage   | Main Question                            | Representative Experiment                                        |   Best LB | Main Finding                                          |
| ------- | ---------------------------------------- | ---------------------------------------------------------------- | --------: | ----------------------------------------------------- |
| Stage 1 | Can we run LoRA SFT successfully?        | `official500_answer_only_baseline`                               | 0.61–0.62 | Answer-only is stable                                 |
| Stage 2 | Can self-generated data help?            | `exp_synth_unit_gravity_answer_only_2000`                        |      0.64 | Solver-backed unit/gravity data is effective          |
| Stage 3 | Does ordinary CoT or raw trace help?     | `official_generated_cot`, `exp_hk_route_b_8192_r16_attn_unembed` |      0.59 | Uncontrolled long reasoning hurts final-answer output |
| Stage 4 | Can clean reasoning improve performance? | `exp_reproduce_085_clean_cot_r32_8192`                           |  **0.85** | Clean CoT + boxed answer achieves high score          |

---

## Final Lessons

1. **Answer-only SFT is a stable starting point.**
   It ensures that the model focuses on final-answer prediction.

2. **Synthetic data must be clean and verifiable.**
   Unit/gravity data worked because the rules and answers were reliable.

3. **Ordinary CoT is risky.**
   Long reasoning without output control can hurt the final-answer format.

4. **Trace data needs loss and format control.**
   Raw trace provides coverage but may overwhelm answer supervision.

5. **Clean CoT is the strongest current direction.**
   It provides reasoning supervision while keeping the final output evaluable.

---

## Next Experiments

| Planned Experiment                            | Goal                                                        | Priority    |
| --------------------------------------------- | ----------------------------------------------------------- | ----------- |
| `exp_reproduce_085_checkpoint_sweep_r32_8192` | Submit intermediate checkpoints to search for 0.86+         | High        |
| `exp_reproduce_085_seed7_r32_8192`            | Seed sweep on the same clean CoT setup                      | Medium      |
| `exp_085_stage2_final_answer_lr1e5_short`     | Short Stage2 final-answer correction from 0.85 adapter      | Medium      |
| `exp_reproduce_085_r32_8192_with_lm_head`     | Add `lm_head` / unembed to improve final token distribution | Low / risky |

---

## Notes

* Do not upload `adapter_model.safetensors`, `submission.zip`, or Kaggle credentials to GitHub.
* Large outputs should be stored as Kaggle outputs or datasets, not in Git.
* Unknown dates, exact trainable parameters, and some debug-generation details are still marked as `TBD`.
