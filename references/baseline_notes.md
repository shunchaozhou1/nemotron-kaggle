# Runnable Baseline Notes

This notebook has successfully run on Kaggle RTX Pro 6000.

Best known configuration:
- MAX_TRAIN_SAMPLES = 500
- MAX_LENGTH = 1024
- RANDOM_STATE = 42
- LORA_RANK = 4
- LORA_ALPHA = 8
- LR = 2e-4
- target_modules = r".*\.(in_proj|out_proj)$"
- LB score = 0.62

Important constraints:
- Do not use trl.
- Do not use SFTTrainer or Trainer.
- Use handwritten PyTorch training loop.
- RTX Pro 6000 has no Internet.
- Must handle cutlass, mamba_ssm, ptxas, and ptxas-blackwell.
- Save adapter to /kaggle/working/adapter.
- Create /kaggle/working/submission.zip.