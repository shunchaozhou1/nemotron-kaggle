
## Parameter Tuning Experiments

Date: 2026-05-19 / 2026-05-20

Current stable baseline:

| Exp | Samples | Max Length | Rank | LR | Dropout | Epochs | Grad Accum | Seed | LB | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline | 500 | 1024 | 4 | 2e-4 | 0.05 | 1 | 4 | 42 | 0.62 | Best current parameter baseline |

Parameter sweep results:

| Exp | Main Change | LB | Conclusion |
|---|---|---:|---|
| exp_seed7 | random_state = 7 | 0.60 | Worse than baseline |
| exp_seed99 | random_state = 99 | 0.59 | Worse than baseline |
| exp_seed2025 | random_state = 2025 | 0.60 | Worse than baseline |
| exp_dropout0 | lora_dropout = 0.0 | 0.61 | Slightly worse than baseline |
| exp_epoch2 | num_epochs = 2 | 0.58 | Overtraining or distribution shift; worse |
| exp_accum2 | grad_accum_steps = 2 | 0.60 | More optimizer updates did not help |
| nemotron-sft-exp002-v3 | CLI reproduced SFT run | 0.59 | Did not exceed baseline |

Summary:

The parameter tuning stage appears to have reached a plateau around LB = 0.62.  
Changing random seed, disabling LoRA dropout, increasing training epochs, and increasing optimizer update frequency did not improve performance.

Current best configuration remains:

```python
MAX_TRAIN_SAMPLES = 500
MAX_LENGTH = 1024
RANDOM_STATE = 42
LORA_RANK = 4
LORA_ALPHA = 8
LORA_DROPOUT = 0.05
LR = 2e-4
GRAD_ACCUM_STEPS = 4
NUM_EPOCHS = 1
TARGET_MODULES = r".*\.(in_proj|out_proj)$"