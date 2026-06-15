import glob
import json
import math
import os
import random
import re
import shutil
import site
import stat
import subprocess
import sys
import time
import zipfile
from collections import defaultdict


os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="strict")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="strict")


TRAIN_ON_KAGGLE = 1
USE_PRETRAINED = 0
assert (TRAIN_ON_KAGGLE + USE_PRETRAINED) == 1

EXP_NAME = "exp_reproduce_085_clean_cot_r32_8192"
BASE_MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

MAX_SEQ_LEN = 8192
LORA_RANK = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
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
NUM_EPOCHS = 1
LEARNING_RATE = 2e-4
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.03
BF16 = True
PACKING = False
SEED = 42

PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 16
DATALOADER_NUM_WORKERS = 2

PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)

OUTPUT_DIR = "/kaggle/working"
ADAPTER_DIR = "/kaggle/working/sft_adapter"
SUBMISSION_ADAPTER_DIR = "/kaggle/working/submission_adapter"
SUBMISSION_ZIP = "/kaggle/working/submission.zip"
PYDEPS_DIR = "/kaggle/working/pydeps"


def print_config():
    cfg = {
        "EXP_NAME": EXP_NAME,
        "TRAIN_ON_KAGGLE": TRAIN_ON_KAGGLE,
        "USE_PRETRAINED": USE_PRETRAINED,
        "MAX_SEQ_LEN": MAX_SEQ_LEN,
        "LORA_RANK": LORA_RANK,
        "LORA_ALPHA": LORA_ALPHA,
        "LORA_DROPOUT": LORA_DROPOUT,
        "TARGET_MODULES": TARGET_MODULES,
        "NUM_EPOCHS": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "lr_scheduler_type": LR_SCHEDULER_TYPE,
        "warmup_ratio": WARMUP_RATIO,
        "bf16": BF16,
        "packing": PACKING,
        "seed": SEED,
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "dataloader_num_workers": DATALOADER_NUM_WORKERS,
    }
    print("Experiment:", EXP_NAME)
    print(json.dumps(cfg, indent=2, ensure_ascii=False))


def list_kaggle_input_summary():
    print("Kaggle input summary:")
    for path in sorted(glob.glob("/kaggle/input/*")):
        print(" -", path)


def find_first_existing(candidates, glob_patterns, description):
    checked = []
    for path in candidates:
        checked.append(path)
        if os.path.exists(path):
            print(f"Resolved {description}: {path}")
            return path
    for pattern in glob_patterns:
        matches = sorted(glob.glob(pattern, recursive=True))
        checked.append(pattern)
        if matches:
            print(f"Resolved {description}: {matches[0]}")
            return matches[0]
    print(f"Missing required {description}. Checked:")
    for item in checked:
        print(" -", item)
    list_kaggle_input_summary()
    raise FileNotFoundError(f"Could not find required {description}")


def recursive_wheels(pattern):
    return sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))


def install_triton_wheel():
    candidates = glob.glob("/kaggle/input/**/*triton*.whl", recursive=True)
    print("Found Triton wheels:", candidates)
    if not candidates:
        raise FileNotFoundError("No Triton wheel found under /kaggle/input")
    wheel = candidates[0]
    os.makedirs(PYDEPS_DIR, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            PYDEPS_DIR,
            "--upgrade",
            "--ignore-installed",
            wheel,
        ],
        check=True,
    )
    if PYDEPS_DIR not in sys.path:
        sys.path.insert(0, PYDEPS_DIR)
    site.addsitedir(PYDEPS_DIR)
    import importlib.util

    print("triton spec:", importlib.util.find_spec("triton"))
    print("✓ Triton wheel installed")


def initialize_blackwell_ptxas():
    sys.path.insert(0, "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script")

    ptxas_src = (
        "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script/"
        "triton/backends/nvidia/bin/ptxas-blackwell"
    )
    ptxas_dst = "/tmp/ptxas-blackwell"
    if os.path.exists(ptxas_src) and not os.path.exists(ptxas_dst):
        shutil.copy2(ptxas_src, ptxas_dst)
        os.chmod(
            ptxas_dst,
            os.stat(ptxas_dst).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH,
        )

        src_bin = os.path.dirname(ptxas_src)
        dst_bin = "/tmp/triton_nvidia_bin"
        shutil.copytree(src_bin, dst_bin, dirs_exist_ok=True)
        for name in os.listdir(dst_bin):
            fp = os.path.join(dst_bin, name)
            if os.path.isfile(fp):
                os.chmod(
                    fp,
                    os.stat(fp).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH,
                )

        os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = ptxas_dst

        import triton.backends.nvidia as nv_backend

        nv_backend.__file__ = os.path.join(dst_bin, "..", "__init__.py")
        os.environ["TRITON_PTXAS_PATH"] = ptxas_dst
    else:
        print("ptxas-blackwell source already missing or copied:", ptxas_src, ptxas_dst)

    import triton.backends.nvidia.compiler as nv_compiler

    nv_compiler.get_ptxas_version = lambda arch: "12.0"
    print("✓ Blackwell ptxas initialized")


def find_packages_dir():
    candidates = [
        "/kaggle/input/datasets/mayukh18/nemotron-packages/packages",
        "/kaggle/input/mayukh18/nemotron-packages/packages",
    ]
    for path in candidates:
        if os.path.isdir(path):
            print("Resolved packages_dir:", path)
            return path
    matches = sorted(glob.glob("/kaggle/input/**/nemotron-packages/**/packages", recursive=True))
    if matches:
        print("Resolved packages_dir:", matches[0])
        return matches[0]
    list_kaggle_input_summary()
    raise FileNotFoundError("Could not find nemotron-packages/packages directory")


def install_training_packages():
    import torch

    packages_dir = find_packages_dir()
    all_mamba = recursive_wheels("mamba_ssm-*.whl")
    all_causal = recursive_wheels("causal*conv1d*.whl")
    print("Found mamba wheels:", all_mamba)
    print("Found causal-conv1d wheels:", all_causal)
    print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
    if not torch.cuda.is_available():
        raise RuntimeError("Need GPU runtime")

    os.makedirs(PYDEPS_DIR, exist_ok=True)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-index",
            "--find-links",
            packages_dir,
            "--target",
            PYDEPS_DIR,
            "--upgrade",
            "unsloth",
            "trl",
            "peft",
            "transformers",
            "datasets",
            "accelerate",
            "bitsandbytes",
        ],
        capture_output=True,
        text=True,
    )
    print("STDOUT:", result.stdout[-1500:])
    print("STDERR:", result.stderr[-1500:])
    print("Return code:", result.returncode)

    if PYDEPS_DIR not in sys.path:
        sys.path.insert(0, PYDEPS_DIR)
    site.addsitedir(PYDEPS_DIR)

    def pick_last(wheels):
        return wheels[-1] if wheels else None

    causal_wheel = pick_last(all_causal)
    mamba_wheel = pick_last(all_mamba)
    print("Selected causal:", causal_wheel)
    print("Selected mamba:", mamba_wheel)
    if causal_wheel:
        r2 = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--target",
                PYDEPS_DIR,
                causal_wheel,
            ],
            capture_output=True,
            text=True,
        )
        print("causal_conv1d:", r2.returncode, r2.stderr[-500:])
    if mamba_wheel:
        r3 = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--target",
                PYDEPS_DIR,
                mamba_wheel,
            ],
            capture_output=True,
            text=True,
        )
        print("mamba_ssm:", r3.returncode, r3.stderr[-500:])
    else:
        raise FileNotFoundError("No mamba_ssm wheel found")
    print("✓ Training packages installed")


def prepare_environment():
    install_triton_wheel()
    initialize_blackwell_ptxas()
    install_training_packages()


def patch_torchcodec_import_for_sentence_transformers():
    """
    Workaround for Kaggle Blackwell environment.

    Unsloth may import transformers / sentence_transformers, which may check or
    import torchcodec. The real torchcodec package fails to load because of
    FFmpeg / libtorchcodec / PyTorch compatibility issues.

    This training script is text-only, so we provide minimal dummy torchcodec
    modules with valid __spec__ fields.
    """
    import sys
    import types
    import importlib.machinery

    # Remove partially imported real torchcodec modules, if any.
    for name in list(sys.modules.keys()):
        if name == "torchcodec" or name.startswith("torchcodec."):
            del sys.modules[name]

    def make_module(name, is_package=False):
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(
            name=name,
            loader=None,
            is_package=is_package,
        )
        if is_package:
            module.__path__ = []
        return module

    torchcodec = make_module("torchcodec", is_package=True)
    torchcodec.__version__ = "0.0.0"

    decoders = make_module("torchcodec.decoders")
    encoders = make_module("torchcodec.encoders")
    samplers = make_module("torchcodec.samplers")
    transforms = make_module("torchcodec.transforms")

    class _DummyDecoder:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "torchcodec dummy decoder is not available in this text-only training script."
            )

    decoders.AudioDecoder = _DummyDecoder
    decoders.VideoDecoder = _DummyDecoder

    torchcodec.decoders = decoders
    torchcodec.encoders = encoders
    torchcodec.samplers = samplers
    torchcodec.transforms = transforms

    sys.modules["torchcodec"] = torchcodec
    sys.modules["torchcodec.decoders"] = decoders
    sys.modules["torchcodec.encoders"] = encoders
    sys.modules["torchcodec.samplers"] = samplers
    sys.modules["torchcodec.transforms"] = transforms


def load_model_and_tokenizer():
    import kagglehub
    import torch
    patch_torchcodec_import_for_sentence_transformers()
    from unsloth import FastLanguageModel

    model_path = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
    print(f"Model path: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        trust_remote_code=True,
        unsloth_force_compile=False,
        attn_implementation="eager",
        dtype=torch.bfloat16,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Model loaded")
    return model, tokenizer


def apply_lora(model):
    patch_torchcodec_import_for_sentence_transformers()
    from unsloth import FastLanguageModel

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )
    model.print_trainable_parameters()
    return model


def resolve_dataset_paths():
    dataset_path = find_first_existing(
        [
            "/kaggle/input/datasets/dgxchen/nemotron-cot-tong/problem_ids_matched.csv",
            "/kaggle/input/dgxchen/nemotron-cot-tong/problem_ids_matched.csv",
        ],
        ["/kaggle/input/**/problem_ids_matched.csv"],
        "dgxchen/nemotron-cot-tong problem_ids_matched.csv",
    )
    augment_path = find_first_existing(
        [
            "/kaggle/input/datasets/amit393/nemotron-cotscv/augmented_examples_v3.csv",
            "/kaggle/input/amit393/nemotron-cotscv/augmented_examples_v3.csv",
        ],
        ["/kaggle/input/**/augmented_examples_v3.csv"],
        "amit393/nemotron-cotscv augmented_examples_v3.csv",
    )
    return dataset_path, augment_path


def boxed_list(s):
    return re.findall(r"\\boxed\{([^}]*)\}", s)


def build_assistant_content(cot, answer):
    boxes = boxed_list(cot)
    answer_str = str(answer).strip()

    if len(boxes) < 2:
        cleaned = re.sub(r"\\boxed\{[^}]*\}", "", cot).rstrip()
        return cleaned + f"\n</think>\n\\boxed{{{answer_str}}}", True

    cot_final = boxes[-1].strip()
    think_close = cot.rfind("</think>")
    body = cot[:think_close].rstrip() if think_close != -1 else cot.rstrip()

    if cot_final == answer_str:
        rounding_note = ""
    else:
        try:
            if abs(float(cot_final) - float(answer_str)) < 1.0:
                rounding_note = (
                    f"\n\nRounding {cot_final} to match the required "
                    f"precision gives {answer_str}."
                )
            else:
                return None, False
        except ValueError:
            return None, False

    assistant_content = body + rounding_note + f"\n</think>\n\\boxed{{{answer_str}}}"
    return assistant_content, True


def load_and_clean_records():
    import pandas as pd
    from datasets import Dataset as HFDataset

    dataset_path, augment_path = resolve_dataset_paths()
    df = pd.read_csv(dataset_path)
    aug_df = pd.read_csv(augment_path)
    print(f"Original (dgxchen): {len(df)} rows")
    print(f"Augmented (yours):  {len(aug_df)} rows")
    df = pd.concat([df, aug_df], ignore_index=True)
    print(f"Combined total:     {len(df)} rows")
    print("\nType distribution after combining:")
    print(df["type"].value_counts().sort_index())

    train_df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    records = []
    record_types = []
    n_dropped = 0
    n_rounding = 0
    for _, row in train_df.iterrows():
        prompt = str(row["prompt"])
        answer = str(row["answer"])
        cot = str(row["generated_cot"])
        if not cot or cot == "nan" or len(cot.strip()) < 5:
            n_dropped += 1
            continue
        before = cot
        assistant_content, ok = build_assistant_content(cot, answer)
        if not ok:
            n_dropped += 1
            continue
        if "Rounding " in assistant_content and before != assistant_content:
            n_rounding += 1
        user_content = prompt + PROMPT_SUFFIX
        records.append(
            {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
            }
        )
        record_types.append(str(row["type"]))

    dataset = HFDataset.from_list(records)
    print(f"\nSFT records: {len(records)} (dropped {n_dropped})")
    print(f"Rounding corrections: {n_rounding}")
    print("Type counts:", dict(sorted(pd.Series(record_types).value_counts().to_dict().items())))

    if records:
        tail = records[0]["messages"][1]["content"][-800:]
        print("Formatted assistant sample tail:")
        print(tail)
        print("contains </think>:", "</think>" in tail)
        print("contains \\boxed{:", "\\boxed{" in tail)
    return dataset, record_types


def build_formatting_func(tokenizer):
    def formatting_prompts_func(example):
        messages = example["messages"]
        if messages and isinstance(messages[0], dict):
            conversations = [messages]
        else:
            conversations = messages
        texts = []
        for conversation in conversations:
            try:
                text = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=True,
                )
            except TypeError:
                text = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            texts.append(text)
        return texts

    return formatting_prompts_func


def build_stratified_index_order(labels, batch_size, seed):
    by_label = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label[label].append(idx)
    rng = random.Random(seed)
    for idx_list in by_label.values():
        rng.shuffle(idx_list)
    n_batches = max(1, math.ceil(len(labels) / batch_size))
    batches = [[] for _ in range(n_batches)]
    batch_order = list(range(n_batches))
    rng.shuffle(batch_order)
    assigned = 0
    for label in sorted(by_label.keys()):
        for idx in by_label[label]:
            batches[batch_order[assigned % n_batches]].append(idx)
            assigned += 1
    return [idx for batch in batches for idx in batch]


def train_and_save(model, tokenizer, dataset, record_types):
    import pandas as pd
    from torch.utils.data import DataLoader, Sampler
    from trl import SFTConfig, SFTTrainer

    training_args = SFTConfig(
        output_dir="/kaggle/working/sft_output",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_ratio=WARMUP_RATIO,
        max_length=MAX_SEQ_LEN,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        weight_decay=0.0,
        max_grad_norm=1e9,
        logging_steps=10,
        save_strategy="no",
        bf16=BF16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        remove_unused_columns=False,
        seed=SEED,
        report_to="none",
        packing=PACKING,
    )

    class PrecomputedOrderSampler(Sampler):
        def __init__(self, order):
            self.order = list(order)

        def __iter__(self):
            return iter(self.order)

        def __len__(self):
            return len(self.order)

    class StratifiedSFTTrainer(SFTTrainer):
        def __init__(self, *args, stratified_order=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.stratified_order = stratified_order

        def get_train_dataloader(self):
            if self.stratified_order is None:
                return super().get_train_dataloader()
            dataloader_kwargs = {
                "batch_size": self.args.per_device_train_batch_size,
                "sampler": PrecomputedOrderSampler(self.stratified_order),
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "persistent_workers": self.args.dataloader_persistent_workers,
                "drop_last": self.args.dataloader_drop_last,
            }
            if self.args.dataloader_num_workers > 0:
                dataloader_kwargs["prefetch_factor"] = self.args.dataloader_prefetch_factor
            return DataLoader(self.train_dataset, **dataloader_kwargs)

    effective_batch_size = max(
        1,
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
    )
    stratified_order = build_stratified_index_order(record_types, effective_batch_size, SEED)
    print(f"Effective batch size: {effective_batch_size}")
    print("Type counts:", dict(sorted(pd.Series(record_types).value_counts().to_dict().items())))
    print("Stratified batching by type enabled")

    trainer = StratifiedSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        formatting_func=build_formatting_func(tokenizer),
        stratified_order=stratified_order,
    )

    print(">>> TRAINING STARTING NOW <<<")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"Training done in {elapsed/60:.1f} min")

    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"✓ Adapter saved to {ADAPTER_DIR}")
    adapter_model = os.path.join(ADAPTER_DIR, "adapter_model.safetensors")
    if os.path.exists(adapter_model):
        print(f"adapter_model.safetensors: {os.path.getsize(adapter_model)/1024/1024:.1f} MB")


def package_submission():
    os.makedirs(SUBMISSION_ADAPTER_DIR, exist_ok=True)
    required_files = ["adapter_config.json", "adapter_model.safetensors"]
    for fname in required_files:
        src = os.path.join(ADAPTER_DIR, fname)
        dst = os.path.join(SUBMISSION_ADAPTER_DIR, fname)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Missing: {src}")
        shutil.copy2(src, dst)
        print(f"Copied {fname} ({os.path.getsize(dst)/1024/1024:.1f} MB)")

    config_path = os.path.join(SUBMISSION_ADAPTER_DIR, "adapter_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["base_model_name_or_path"] = BASE_MODEL_NAME
    cfg["inference_mode"] = True
    cfg["lora_dropout"] = 0.0
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    if os.path.exists(SUBMISSION_ZIP):
        os.remove(SUBMISSION_ZIP)
    with zipfile.ZipFile(SUBMISSION_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in required_files:
            zf.write(os.path.join(SUBMISSION_ADAPTER_DIR, fname), fname)
            print(f"  Added {fname}")
    print(f"\nsubmission.zip: {os.path.getsize(SUBMISSION_ZIP)/1024/1024:.1f} MB")
    print("✓ Ready to submit!")


def main():
    print_config()
    if TRAIN_ON_KAGGLE:
        prepare_environment()
        model, tokenizer = load_model_and_tokenizer()
        model = apply_lora(model)
        dataset, record_types = load_and_clean_records()
        train_and_save(model, tokenizer, dataset, record_types)
        package_submission()
    else:
        raise RuntimeError("This reproduction script is intended for Kaggle execution.")


if __name__ == "__main__":
    main()
