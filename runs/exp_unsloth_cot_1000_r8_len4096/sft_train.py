import glob
import json
import os
import random
import re
import shutil
import site
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"

EXP_NAME = "exp_unsloth_cot_1000_r8_len4096"
SEED = 42
NUM_COT_SAMPLES = 1000
MAX_SEQ_LEN = 4096
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LR = 2e-4
NUM_EPOCHS = 1
PER_DEVICE_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16

COT_DATASET_PATH = "/kaggle/input/datasets/dgxchen/nemotron-cot-tong/problem_ids_matched.csv"
MODEL_HANDLE = "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
BASE_MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
ADAPTER_DIR = "/kaggle/working/sft_adapter"
ZIP_PATH = "/kaggle/working/submission.zip"
PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj",
    "out_proj",
    "up_proj",
    "down_proj",
]


def log_experiment_config():
    print("EXP_NAME:", EXP_NAME)
    print("SEED:", SEED)
    print("NUM_COT_SAMPLES=1000")
    print("MAX_SEQ_LEN=4096")
    print("LORA_RANK=8")
    print("LORA_ALPHA:", LORA_ALPHA)
    print("LORA_DROPOUT:", LORA_DROPOUT)
    print("LR:", LR)
    print("NUM_EPOCHS:", NUM_EPOCHS)
    print("PER_DEVICE_BATCH_SIZE:", PER_DEVICE_BATCH_SIZE)
    print("GRAD_ACCUM_STEPS:", GRAD_ACCUM_STEPS)


def initialize_kaggle_environment():
    utility_paths = [
        "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script",
        "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script",
    ]
    for path in utility_paths:
        if os.path.exists(path):
            site.addsitedir(path)
            sys.path.insert(0, path)
            print("Added utility path:", path)

    cutlass_paths = glob.glob(
        "/kaggle/usr/lib/notebooks/ryanholbrook/**/nvidia_cutlass_dsl/python_packages",
        recursive=True,
    )
    for path in cutlass_paths:
        if os.path.exists(path):
            site.addsitedir(path)
            sys.path.insert(0, path)
            print("Added CUTLASS path:", path)

    mamba_paths = glob.glob(
        "/kaggle/usr/lib/notebooks/ryanholbrook/**/mamba_ssm",
        recursive=True,
    )
    for path in mamba_paths:
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
            dst = f"/tmp/{name}"
            shutil.copy(found[0], dst)
            os.chmod(dst, 0o755)
            if name == "ptxas":
                os.environ["TRITON_PTXAS_PATH"] = dst
            else:
                os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = dst
            print(f"Prepared {name}:", dst)

    os.environ["PATH"] = "/tmp:" + os.environ.get("PATH", "")
    print("TRITON_PTXAS_PATH =", os.environ.get("TRITON_PTXAS_PATH"))
    print(
        "TRITON_PTXAS_BLACKWELL_PATH =",
        os.environ.get("TRITON_PTXAS_BLACKWELL_PATH"),
    )


def patch_triton_ptxas_version():
    try:
        import triton.backends.nvidia.compiler as nvidia_compiler

        nvidia_compiler.get_ptxas_version = lambda arch: "12.0"
        print("Patched triton.backends.nvidia.compiler.get_ptxas_version")
    except Exception as exc:
        print("Could not patch Triton ptxas version:", repr(exc))


def install_offline_packages():
    package_names = [
        "unsloth",
        "trl",
        "peft",
        "transformers",
        "datasets",
        "accelerate",
        "bitsandbytes",
        "causal_conv1d",
        "mamba_ssm",
    ]
    roots = [
        "/kaggle/input/datasets/mayukh18/nemotron-packages",
        "/kaggle/input/datasets/dennisfong/nvidia-nemotron-offline-packages",
        "/kaggle/input/datasets",
    ]
    existing_roots = [root for root in roots if os.path.exists(root)]
    wheel_paths = []
    find_links = set()

    for root in existing_roots:
        for wheel in glob.glob(os.path.join(root, "**", "*.whl"), recursive=True):
            wheel_paths.append(wheel)
            find_links.add(os.path.dirname(wheel))
        for package_name in package_names:
            for path in glob.glob(os.path.join(root, "**", package_name), recursive=True):
                if os.path.isdir(path):
                    sys.path.insert(0, os.path.dirname(path))
                    print("Added offline package directory to sys.path:", os.path.dirname(path))

    print("Offline wheel count:", len(wheel_paths))
    for directory in sorted(find_links)[:30]:
        print("Offline find-links:", directory)

    if not wheel_paths and not find_links:
        print("No offline wheels found; continuing with preinstalled packages.")
        return

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--upgrade",
    ]
    for directory in sorted(find_links):
        cmd.extend(["--find-links", directory])
    cmd.extend(package_names)
    print("Installing offline packages with --no-index")
    subprocess.run(cmd, check=False)


def clean_cot_text(text):
    text = str(text)
    text = re.sub(r"\\boxed\{[^{}]*\}", "", text)
    return text.rstrip()


def load_cot_records():
    import pandas as pd

    if not os.path.exists(COT_DATASET_PATH):
        print("CoT dataset not found at:", COT_DATASET_PATH)
        print("Candidate files under /kaggle/input/datasets:")
        for path in glob.glob("/kaggle/input/datasets/**/*", recursive=True)[:300]:
            print(path)
        raise FileNotFoundError(COT_DATASET_PATH)

    df = pd.read_csv(COT_DATASET_PATH)
    print("Full CoT dataset rows:", len(df))
    required_columns = {"prompt", "answer", "generated_cot", "type"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"CoT dataset is missing columns: {missing}")

    valid = df.dropna(subset=["generated_cot"]).copy()
    valid["generated_cot"] = valid["generated_cot"].astype(str)
    valid = valid[valid["generated_cot"].str.strip().str.len() >= 5].copy()
    valid["cot_cleaned"] = valid["generated_cot"].map(clean_cot_text)
    valid = valid[valid["cot_cleaned"].str.strip().str.len() >= 5].copy()
    print("Valid CoT rows:", len(valid))

    if len(valid) < NUM_COT_SAMPLES:
        raise ValueError(
            f"Requested {NUM_COT_SAMPLES} CoT samples, but only {len(valid)} are valid."
        )

    sampled = valid.sample(n=NUM_COT_SAMPLES, random_state=SEED).reset_index(drop=True)
    records = []
    for _, row in sampled.iterrows():
        user_content = str(row["prompt"]).rstrip() + PROMPT_SUFFIX
        assistant_content = (
            str(row["cot_cleaned"]).rstrip()
            + "\n</think>\n\\boxed{"
            + str(row["answer"]).strip()
            + "}"
        )
        records.append(
            {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ],
                "type": str(row["type"]),
            }
        )

    print(f"SFT records: {len(records)}")
    print("type distribution:")
    print(sampled["type"].value_counts(dropna=False).to_string())
    return records


def formatting_prompts_func(example, tokenizer):
    try:
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )


def build_stratified_index_order(labels, batch_size, seed):
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for index, label in enumerate(labels):
        buckets[str(label)].append(index)
    for indices in buckets.values():
        rng.shuffle(indices)

    ordered = []
    keys = sorted(buckets)
    while any(buckets[key] for key in keys):
        batch = []
        active_keys = [key for key in keys if buckets[key]]
        rng.shuffle(active_keys)
        while len(batch) < batch_size and active_keys:
            for key in list(active_keys):
                if buckets[key] and len(batch) < batch_size:
                    batch.append(buckets[key].pop())
                if not buckets[key]:
                    active_keys.remove(key)
        ordered.extend(batch)
    return ordered


class PrecomputedOrderSampler:
    def __init__(self, order):
        self.order = list(order)

    def __iter__(self):
        return iter(self.order)

    def __len__(self):
        return len(self.order)


def build_trainer_class():
    from torch.utils.data import DataLoader
    from trl import SFTTrainer

    class StratifiedSFTTrainer(SFTTrainer):
        def __init__(self, *args, type_labels=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.type_labels = list(type_labels or [])

        def get_train_dataloader(self):
            if self.train_dataset is None:
                raise ValueError("Trainer: training requires a train_dataset.")
            effective_batch_size = PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS
            labels = self.type_labels[: len(self.train_dataset)]
            order = build_stratified_index_order(labels, effective_batch_size, SEED)
            sampler = PrecomputedOrderSampler(order)
            dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
            print("Stratified batching by type enabled")
            return self.accelerator.prepare(dataloader)

    return StratifiedSFTTrainer


def load_model_and_tokenizer():
    import kagglehub
    import torch
    from unsloth import FastLanguageModel

    model_path = kagglehub.model_download(MODEL_HANDLE)
    print("Model path:", model_path)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LEN,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        trust_remote_code=True,
        attn_implementation="eager",
        unsloth_force_compile=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )
    return model, tokenizer


def train_and_save(model, tokenizer, records):
    from datasets import Dataset
    from trl import SFTConfig

    train_dataset = Dataset.from_list(records)
    type_labels = list(train_dataset["type"])

    def map_to_text(example):
        return {"text": formatting_prompts_func(example, tokenizer)}

    train_dataset = train_dataset.map(map_to_text, remove_columns=["messages"])
    sample_text = train_dataset[0]["text"]
    print("apply_chat_template sample tail:")
    print(sample_text[-1200:])
    print("contains </think>:", "</think>" in sample_text)
    print("contains \\boxed{}:", "\\boxed{" in sample_text)

    sft_config_kwargs = {
        "output_dir": "/kaggle/working/sft_output",
        "num_train_epochs": NUM_EPOCHS,
        "per_device_train_batch_size": PER_DEVICE_BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "learning_rate": LR,
        "lr_scheduler_type": "linear",
        "warmup_steps": 0,
        "max_length": MAX_SEQ_LEN,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_epsilon": 1e-8,
        "weight_decay": 0.0,
        "max_grad_norm": 1e9,
        "logging_steps": 10,
        "save_strategy": "no",
        "bf16": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "dataloader_num_workers": 2,
        "remove_unused_columns": False,
        "seed": SEED,
        "report_to": "none",
        "packing": False,
    }
    try:
        sft_config = SFTConfig(**sft_config_kwargs, dataset_text_field="text")
    except TypeError:
        sft_config = SFTConfig(**sft_config_kwargs)

    StratifiedSFTTrainer = build_trainer_class()
    trainer_kwargs = {
        "model": model,
        "args": sft_config,
        "train_dataset": train_dataset,
        "type_labels": type_labels,
    }
    trainer = None
    trainer_attempts = [
        {"processing_class": tokenizer},
        {"tokenizer": tokenizer},
        {"processing_class": tokenizer, "dataset_text_field": "text"},
        {"tokenizer": tokenizer, "dataset_text_field": "text"},
    ]
    for extra_kwargs in trainer_attempts:
        try:
            trainer = StratifiedSFTTrainer(**trainer_kwargs, **extra_kwargs)
            break
        except TypeError as exc:
            print("SFTTrainer init attempt failed:", repr(exc))
    if trainer is None:
        raise TypeError("Could not initialize SFTTrainer with available signatures.")

    trainer.train()
    trainer.model.save_pretrained(ADAPTER_DIR)
    print("Saved adapter to:", ADAPTER_DIR)


def patch_adapter_config():
    adapter_config_path = os.path.join(ADAPTER_DIR, "adapter_config.json")
    with open(adapter_config_path, "r", encoding="utf-8") as f:
        adapter_config = json.load(f)
    adapter_config["base_model_name_or_path"] = BASE_MODEL_NAME
    adapter_config["inference_mode"] = True
    adapter_config["lora_dropout"] = 0.0
    with open(adapter_config_path, "w", encoding="utf-8") as f:
        json.dump(adapter_config, f, indent=2, ensure_ascii=False)
    print("Patched adapter_config.json")


def create_submission_zip():
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)

    allowed_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "README.md",
    ]
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in allowed_files:
            path = os.path.join(ADAPTER_DIR, name)
            if os.path.exists(path):
                zf.write(path, arcname=name)

    print("submission.zip created:", ZIP_PATH)
    print("submission.zip size:", os.path.getsize(ZIP_PATH))
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        print("submission.zip contents:")
        for info in zf.infolist():
            print(f" - {info.filename}: {info.file_size} bytes")


def main():
    log_experiment_config()
    random.seed(SEED)
    initialize_kaggle_environment()
    install_offline_packages()
    patch_triton_ptxas_version()
    records = load_cot_records()
    model, tokenizer = load_model_and_tokenizer()
    train_and_save(model, tokenizer, records)
    patch_adapter_config()
    create_submission_zip()


if __name__ == "__main__":
    main()
