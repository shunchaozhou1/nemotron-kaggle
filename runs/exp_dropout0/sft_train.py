import glob
import json
import os
import re
import shutil
import site
import subprocess
import sys

import pandas as pd
import polars as pl
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


TRAIN_PATH_CANDIDATES = [
    "/kaggle/input/nvidia-nemotron-model-reasoning-challenge/train.csv",
    "/kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv",
]

def find_train_csv():
    for path in TRAIN_PATH_CANDIDATES:
        if os.path.exists(path):
            print(f"Found train.csv at: {path}")
            return path

    print("Could not find train.csv in known paths.")
    print("Listing /kaggle/input:")
    for p in glob.glob("/kaggle/input/**/*", recursive=True)[:200]:
        print(p)

    matches = glob.glob("/kaggle/input/**/train.csv", recursive=True)
    if matches:
        print(f"Found train.csv by glob: {matches[0]}")
        return matches[0]

    raise FileNotFoundError("train.csv not found under /kaggle/input")

def load_train_data():
    train_path = find_train_csv()
    train = pl.read_csv(train_path)
    print("Train shape:", train.shape)
    print(train.head())
    return train
MODEL_PATH = "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1"
ADAPTER_DIR = "/kaggle/working/adapter"
SUBMISSION_ZIP = "/kaggle/working/submission.zip"

DEFAULT_CONFIG = {
    "exp_name": "best_baseline",
    "max_train_samples": 500,
    "max_length": 1024,
    "random_state": 42,
    "lora_rank": 4,
    "lora_alpha": 8,
    "lora_dropout": 0.05,
    "lr": 2e-4,
    "target_modules": r".*\.(in_proj|out_proj)$",
    "batch_size": 1,
    "grad_accum_steps": 4,
    "num_epochs": 1,
    "data_mode": "official_only",
    "num_cot_samples": 0,
    "cot_dataset_path": "/kaggle/input/datasets/dgxchen/nemotron-cot-tong/problem_ids_matched.csv",
    "cot_prompt_suffix": "\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`",
}

RUN_CONFIG = {
    "exp_name": "exp_dropout0",
    "lora_dropout": 0.0,
}


def load_config():
    cfg = DEFAULT_CONFIG.copy()
    run_config = globals().get("RUN_CONFIG", {})

    if run_config:
        print("Using embedded RUN_CONFIG from sft_train.py")
        cfg.update(run_config)
    else:
        print("RUN_CONFIG is empty. Using DEFAULT_CONFIG.")

    print("Experiment:", cfg["exp_name"])
    print("Config:")
    print(json.dumps(cfg, indent=2, ensure_ascii=False, sort_keys=True))

    if cfg.get("data_mode") in ["official_plus_cot", "cot_only"]:
        if cfg.get("num_cot_samples", 0) <= 0:
            raise ValueError(
                f"data_mode={cfg.get('data_mode')} requires num_cot_samples > 0."
            )

    return cfg


def initialize_kaggle_environment():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"

    base_candidates = [
        "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script",
        "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script",
    ]

    for base in base_candidates:
        if os.path.exists(base):
            site.addsitedir(base)
            sys.path.insert(0, base)
            print("Added utility path:", base)

    cutlass_candidates = glob.glob(
        "/kaggle/usr/lib/notebooks/ryanholbrook/**/nvidia_cutlass_dsl/python_packages",
        recursive=True,
    )
    for path in cutlass_candidates:
        if os.path.exists(path):
            site.addsitedir(path)
            sys.path.insert(0, path)
            print("Added CUTLASS path:", path)

    mamba_candidates = glob.glob(
        "/kaggle/usr/lib/notebooks/ryanholbrook/**/mamba_ssm",
        recursive=True,
    )
    for path in mamba_candidates:
        parent = os.path.dirname(path)
        if os.path.exists(parent):
            site.addsitedir(parent)
            sys.path.insert(0, parent)
            print("Added mamba_ssm path:", parent)

    for name in ["ptxas", "ptxas-blackwell"]:
        found = glob.glob(
            f"/kaggle/usr/lib/notebooks/ryanholbrook/**/triton/backends/nvidia/bin/{name}",
            recursive=True,
        )

        if found:
            src = found[0]
            dst = f"/tmp/{name}"
            shutil.copy(src, dst)
            os.chmod(dst, 0o755)

            if name == "ptxas":
                os.environ["TRITON_PTXAS_PATH"] = dst
            else:
                os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = dst

            print(f"Prepared {name}:", dst)

    os.environ["PATH"] = "/tmp:" + os.environ.get("PATH", "")

    print("Environment initialized.")
    print("TRITON_PTXAS_PATH =", os.environ.get("TRITON_PTXAS_PATH"))
    print(
        "TRITON_PTXAS_BLACKWELL_PATH =",
        os.environ.get("TRITON_PTXAS_BLACKWELL_PATH"),
    )


def show_binary_version(path):
    if os.path.exists(path):
        subprocess.run([path, "--version"], check=False)


class PromptCompletionDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=1024):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if "train_prompt" in self.df.columns and "train_completion" in self.df.columns:
            prompt = str(row["train_prompt"])
            answer = str(row["train_completion"])
        else:
            prompt = str(row["prompt"]).rstrip() + "\n"
            answer = str(row["answer"]).strip() + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
        )["input_ids"]

        answer_ids = self.tokenizer(
            answer,
            add_special_tokens=False,
        )["input_ids"]

        if len(answer_ids) >= self.max_length:
            input_ids = answer_ids[: self.max_length]
            labels = input_ids.copy()
        else:
            max_prompt_len = self.max_length - len(answer_ids)
            prompt_ids = prompt_ids[-max_prompt_len:]

            input_ids = prompt_ids + answer_ids
            labels = [-100] * len(prompt_ids) + answer_ids.copy()

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def make_collate_fn(tokenizer):
    def collate_fn(batch):
        max_len = max(len(x["input_ids"]) for x in batch)

        input_ids = []
        attention_mask = []
        labels = []

        for x in batch:
            pad_len = max_len - len(x["input_ids"])

            input_ids.append(x["input_ids"] + [tokenizer.pad_token_id] * pad_len)
            attention_mask.append(x["attention_mask"] + [0] * pad_len)
            labels.append(x["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate_fn


def list_kaggle_input_candidates():
    print("Listing candidate files under /kaggle/input:")
    for pattern in [
        "/kaggle/input/**/*.csv",
        "/kaggle/input/**/*.json",
        "/kaggle/input/**/*.parquet",
    ]:
        for path in glob.glob(pattern, recursive=True)[:200]:
            print(path)


def clean_cot_text(text):
    text = str(text)
    text = re.sub(r"\\boxed\{[^{}]*\}", "", text)
    return text.rstrip()


def prepare_official_train_df(train, tokenizer, cfg):
    df = train.to_pandas().sample(
        n=min(cfg["max_train_samples"], train.shape[0]),
        random_state=cfg["random_state"],
    ).reset_index(drop=True)

    df["train_prompt"] = df["prompt"].astype(str).str.rstrip() + "\n"
    df["train_completion"] = df["answer"].astype(str).str.strip() + tokenizer.eos_token
    df["data_source"] = "official"
    return df


def load_cot_train_df(tokenizer, cfg):
    cot_path = cfg["cot_dataset_path"]
    if not os.path.exists(cot_path):
        list_kaggle_input_candidates()
        raise FileNotFoundError(
            f"CoT dataset is required for data_mode={cfg['data_mode']!r}, "
            f"but cot_dataset_path does not exist: {cot_path}"
        )

    df = pl.read_csv(cot_path).to_pandas()
    required_columns = {"prompt", "answer", "generated_cot", "type"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"CoT dataset is missing columns: {missing_columns}")

    cot = df.dropna(subset=["generated_cot"]).copy()
    cot["generated_cot"] = cot["generated_cot"].astype(str)
    cot = cot[cot["generated_cot"].str.strip().str.len() >= 5].copy()
    cot["cot_cleaned"] = cot["generated_cot"].map(clean_cot_text)
    cot = cot[cot["cot_cleaned"].str.strip().str.len() >= 5].copy()

    n = min(cfg["num_cot_samples"], len(cot))
    cot = cot.sample(n=n, random_state=cfg["random_state"]).reset_index(drop=True)

    cot["train_prompt"] = cot["prompt"].astype(str) + cfg["cot_prompt_suffix"] + "\n"
    cot["train_completion"] = (
        cot["cot_cleaned"]
        + "\n</think>\n\\boxed{"
        + cot["answer"].astype(str).str.strip()
        + "}"
        + tokenizer.eos_token
    )
    cot["data_source"] = "cot"
    return cot


def prepare_training_data(train, tokenizer, cfg):
    data_mode = cfg["data_mode"]
    if data_mode not in {"official_only", "cot_only", "official_plus_cot"}:
        raise ValueError(
            "data_mode must be one of: official_only, cot_only, official_plus_cot"
        )

    official_df = None
    cot_df = None

    if data_mode in {"official_only", "official_plus_cot"}:
        official_df = prepare_official_train_df(train, tokenizer, cfg)

    if data_mode in {"cot_only", "official_plus_cot"}:
        cot_df = load_cot_train_df(tokenizer, cfg)

    if data_mode == "official_only":
        train_df = official_df
    elif data_mode == "cot_only":
        train_df = cot_df.sample(frac=1, random_state=cfg["random_state"]).reset_index(
            drop=True
        )
    else:
        train_df = (
            pd.concat([official_df, cot_df], ignore_index=True, sort=False)
            .sample(frac=1, random_state=cfg["random_state"])
            .reset_index(drop=True)
        )

    official_count = int((train_df["data_source"] == "official").sum())
    cot_count = int((train_df["data_source"] == "cot").sum())
    print(
        "Training data counts: "
        f"total={len(train_df)}, official={official_count}, cot={cot_count}"
    )

    if data_mode in {"official_plus_cot", "cot_only"} and cot_count == 0:
        raise RuntimeError(
            f"data_mode={data_mode} requires CoT samples, but prepared cot samples = 0."
        )

    return train_df



def load_model_and_tokenizer():
    print("CUDA device count:", torch.cuda.device_count())
    print("Device name:", torch.cuda.get_device_name(0))
    print("BF16 supported:", torch.cuda.is_bf16_supported())

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print("Using dtype:", dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map={"": 0},
            trust_remote_code=True,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map={"": 0},
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

    print("Model loaded.")
    return model, tokenizer, dtype


def apply_lora(model, cfg):
    lora_config = LoraConfig(
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["target_modules"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def train_model(model, tokenizer, dtype, train, cfg):
    model.config.use_cache = False
    model.train()

    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    train_df = prepare_training_data(train, tokenizer, cfg)

    train_dataset = PromptCompletionDataset(
        train_df,
        tokenizer,
        max_length=cfg["max_length"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer),
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"],
    )

    optimizer.zero_grad()

    for epoch in range(1, cfg["num_epochs"] + 1):
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to("cuda:0") for k, v in batch.items()}

            with torch.amp.autocast("cuda", dtype=dtype):
                outputs = model(**batch)
                loss = outputs.loss / cfg["grad_accum_steps"]

            loss.backward()

            if step % cfg["grad_accum_steps"] == 0 or step == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            print(
                f"epoch {epoch}/{cfg['num_epochs']} | "
                f"step {step}/{len(train_loader)} | "
                f"loss = {loss.item() * cfg['grad_accum_steps']:.4f}"
            )

    print("Training finished.")


def save_adapter_and_submission(model):
    shutil.rmtree(ADAPTER_DIR, ignore_errors=True)
    os.makedirs(ADAPTER_DIR, exist_ok=True)

    model.save_pretrained(ADAPTER_DIR)
    print("Saved adapter to:", ADAPTER_DIR)
    subprocess.run(["ls", "-lh", ADAPTER_DIR], check=False)

    if os.path.exists(SUBMISSION_ZIP):
        os.remove(SUBMISSION_ZIP)

    subprocess.run(
        "cd /kaggle/working/adapter && zip -r /kaggle/working/submission.zip .",
        shell=True,
        check=True,
    )

    print("Created:", SUBMISSION_ZIP)
    subprocess.run(["ls", "-lh", SUBMISSION_ZIP], check=False)
    subprocess.run(["unzip", "-l", SUBMISSION_ZIP], check=False)


def main():
    cfg = load_config()
    initialize_kaggle_environment()
    show_binary_version("/tmp/ptxas")
    show_binary_version("/tmp/ptxas-blackwell")

    train = load_train_data()
    model, tokenizer, dtype = load_model_and_tokenizer()
    model = apply_lora(model, cfg)
    train_model(model, tokenizer, dtype, train, cfg)
    save_adapter_and_submission(model)
    print("Done.")


if __name__ == "__main__":
    main()
