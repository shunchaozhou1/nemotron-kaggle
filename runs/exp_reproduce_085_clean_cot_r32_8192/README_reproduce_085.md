# exp_reproduce_085_clean_cot_r32_8192

## Goal

Strong reproduction of `nvidia-nemotron-score-0-85.ipynb`.

This run intentionally keeps the notebook method as-is:

- Unsloth `FastLanguageModel`
- TRL `SFTTrainer` / `SFTConfig`
- clean CoT construction
- chat template with `enable_thinking=True`
- type-stratified sampler
- rank32 LoRA
- max sequence length 8192

No method redesign is intended in this run.

## Kaggle Input Sources

Required:

- `dgxchen/nemotron-cot-tong`
  - expected file: `problem_ids_matched.csv`
- `amit393/nemotron-cotscv`
  - expected file: `augmented_examples_v3.csv`
- `mayukh18/nemotron-packages`
  - expected directory: `packages`
  - used for offline `unsloth`, `trl`, `peft`, `transformers`, `datasets`, `accelerate`, `bitsandbytes`
- `ryanholbrook/nvidia-utility-script`
  - used for Blackwell `ptxas`
- model source:
  - `metric/nemotron-3-nano-30b-a3b-bf16/Transformers/default/1`

The script auto-detects the CSV paths and prints checked paths if a source is missing.

## Key Parameters

| Parameter | Value |
|---|---|
| `TRAIN_ON_KAGGLE` | `1` |
| `USE_PRETRAINED` | `0` |
| `MAX_SEQ_LEN` | `8192` |
| `LORA_RANK` | `32` |
| `LORA_ALPHA` | `32` |
| `LORA_DROPOUT` | `0.0` |
| `target_modules` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj`, `out_proj`, `up_proj`, `down_proj` |
| `NUM_EPOCHS` | `1` |
| `learning_rate` | `2e-4` |
| `lr_scheduler_type` | `cosine` |
| `warmup_ratio` | `0.03` |
| `bf16` | `True` |
| `packing` | `False` |
| `seed` | `42` |
| `per_device_train_batch_size` | `2` |
| `gradient_accumulation_steps` | `16` |

Allowed engineering fallback only if necessary:

- If dataloader worker crashes, change `dataloader_num_workers` from `2` to `0`.
- If OOM, change `per_device_train_batch_size` from `2` to `1`.
- Do not change rank32, max_length8192, target modules, or clean CoT logic for this reproduction.

## Clean CoT Logic

The script preserves the notebook's core cleaning:

1. Extract all `\boxed{...}` payloads from `generated_cot`.
2. Keep the reasoning body before `</think>`.
3. If CoT final equals dataset `answer`, keep it.
4. If numeric values differ only by rounding, append a short rounding correction.
5. If the final answer truly conflicts, drop the row.
6. Always end assistant content with:

```text
</think>
\boxed{answer}
```

## Training Log

To fill after Kaggle run:

- Original rows: TBD
- Augmented rows: TBD
- Combined rows: TBD
- SFT records: TBD
- Dropped rows: TBD
- Rounding corrections: TBD
- Effective batch size: TBD
- Type distribution: TBD
- Training time: TBD

## Runtime / OOM

- OOM: TBD
- If OOM occurred, what was changed: TBD
- Dataloader worker issue: TBD

## Output

Expected outputs:

- adapter: `/kaggle/working/sft_adapter`
- submission: `/kaggle/working/submission.zip`

To fill after Kaggle run:

- `adapter_model.safetensors` size: TBD
- `submission.zip` size: TBD

## LB Score

- Kaggle Public LB: TBD

## Notes

This run intentionally uses `SFTTrainer`, Unsloth, chat template, and long CoT because it is a reproduction of the uploaded 0.85 notebook, unlike the project's hand-written PyTorch answer-only baseline experiments.
