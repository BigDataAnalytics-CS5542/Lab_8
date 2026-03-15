"""
QLoRA Fine-Tuning Script for Legal Demand Assistant
====================================================
Fine-tunes Qwen3.5-9B on a 50-example legal instruction dataset using QLoRA.
Produces adapter weights for later inference in the FastAPI pipeline.

Usage (Colab Pro / A100):
    pip install transformers>=4.46.0 peft>=0.13.0 bitsandbytes>=0.44.0 trl>=0.12.0 datasets accelerate flash-attn
    python qlora_finetune.py
"""

# ── Section 1: Imports and Configuration ─────────────────────────────────────

import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# ── Hyperparameters ──────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen3.5-9B"
OUTPUT_DIR = "./qlora-legal-demand-adapter"
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "instruction_dataset.json")

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
LR_SCHEDULER = "cosine"
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
MAX_SEQ_LENGTH = 1024
WEIGHT_DECAY = 0.01

SYSTEM_PROMPT = (
    "You are a legal demand assistant. You help draft demand letters, "
    "identify legal claims, extract elements from demand letters, "
    "evaluate letter quality, and recommend remedies."
)

# ── Section 2: Load and Format Dataset ───────────────────────────────────────

def load_dataset(data_path: str) -> Dataset:
    """Load instruction dataset and format into ChatML messages."""
    with open(data_path) as f:
        raw_data = json.load(f)

    formatted = []
    for example in raw_data:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{example['instruction']}\n\n{example['input']}"},
            {"role": "assistant", "content": example["output"]},
        ]
        formatted.append({
            "messages": messages,
            "task_type": example["task_type"],
        })

    return Dataset.from_list(formatted)


def split_dataset(dataset: Dataset):
    """Stratified 90/10 split (45 train / 5 val) by task_type."""
    task_types = dataset["task_type"]
    indices = list(range(len(dataset)))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=5,
        random_state=42,
        stratify=task_types,
    )

    train_ds = dataset.select(train_idx)
    val_ds = dataset.select(val_idx)

    # Drop task_type column — not needed for training
    train_ds = train_ds.remove_columns(["task_type"])
    val_ds = val_ds.remove_columns(["task_type"])

    print(f"Train: {len(train_ds)} examples, Val: {len(val_ds)} examples")
    return train_ds, val_ds


# ── Section 3: Load Model and Tokenizer (4-bit) ─────────────────────────────

def load_model_and_tokenizer():
    """Load Qwen3.5-9B in 4-bit quantization with flash attention."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        attn_implementation="flash_attention_2",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    return model, tokenizer


# ── Section 4: Configure LoRA ────────────────────────────────────────────────

def get_lora_config() -> LoraConfig:
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
        bias="none",
    )


# ── Section 5: Configure Training ───────────────────────────────────────────

def get_training_config() -> SFTConfig:
    return SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        weight_decay=WEIGHT_DECAY,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_seq_length=MAX_SEQ_LENGTH,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=5,
        report_to="none",
        seed=42,
    )


# ── Section 6 & 7: Train and Save ───────────────────────────────────────────

def train():
    """Run QLoRA fine-tuning and save adapter weights."""
    data_path = str(Path(DATA_PATH).resolve())
    print(f"Loading dataset from {data_path}")
    dataset = load_dataset(data_path)
    train_ds, val_ds = split_dataset(dataset)

    print(f"Loading model: {MODEL_ID}")
    model, tokenizer = load_model_and_tokenizer()

    peft_config = get_lora_config()
    training_config = get_training_config()

    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print("Starting training...")
    result = trainer.train()

    print("\n── Training Results ──")
    print(f"  Total steps:    {result.global_step}")
    print(f"  Training loss:  {result.training_loss:.4f}")
    for key, value in result.metrics.items():
        print(f"  {key}: {value}")

    # Save adapter and tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nAdapter saved to {OUTPUT_DIR}")

    return model, tokenizer


# ── Section 8: Inference Test ────────────────────────────────────────────────

def run_inference_test():
    """Reload base model + adapter and test with sample prompts."""
    print("\n── Inference Test ──")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        attn_implementation="flash_attention_2",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    model.eval()

    test_prompts = [
        {
            "task": "Demand Letter",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    "Draft a legal demand letter based on the scenario.\n\n"
                    "A dog walker lost a client's dog due to negligence. The dog was "
                    "found two days later, but the owner incurred $1,200 in search and "
                    "veterinary costs."
                )},
            ],
        },
        {
            "task": "Claim Identification",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    "Identify the strongest legal claim based on the scenario.\n\n"
                    "A mechanic charged $800 for car repairs that were never actually performed."
                )},
            ],
        },
        {
            "task": "Element Extraction",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    "Extract the key legal elements from this demand letter.\n\n"
                    "Dear QuickFix Plumbing, I represent Maria Santos regarding "
                    "water damage totaling $3,600 caused by your faulty pipe installation "
                    "on February 15, 2026. Please remit payment within 14 days."
                )},
            ],
        },
    ]

    for prompt in test_prompts:
        text = tokenizer.apply_chat_template(
            prompt["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        print(f"\n{'='*60}")
        print(f"Task: {prompt['task']}")
        print(f"{'='*60}")
        print(response.strip())


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
    run_inference_test()
