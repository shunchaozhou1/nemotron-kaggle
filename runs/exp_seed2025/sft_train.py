import glob
import json
import os
import shutil
import site
import subprocess
import sys

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
}


def load_config():
    cfg = DEFAULT_CONFIG.copy()
    config_path = os.path.join(os.getcwd(), "config.json")

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            overrides = json.load(f)
        cfg.update(overrides)
        print("Loaded config from:", config_path)
    else:
        print("config.json not found. Using DEFAULT_CONFIG.")

    print("Experiment:", cfg["exp_name"])
    print("Config:")
    print(json.dumps(cfg, indent=2, sort_keys=True))
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

    train_pd_small = train.to_pandas().sample(
        n=min(cfg["max_train_samples"], train.shape[0]),
        random_state=cfg["random_state"],
    ).reset_index(drop=True)

    train_dataset = PromptCompletionDataset(
        train_pd_small,
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
