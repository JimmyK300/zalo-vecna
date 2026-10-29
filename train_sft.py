"""
DataMate — Zalo AI Challenge 2026
Supervised Fine-Tuning (SFT) with LoRA / QLoRA

Trains Qwen2.5-Coder-7B on verified Golden Trajectories from build_training_dataset.py.
Outputs LoRA adapter weights ready for inference or GRPO warm-start.
"""

import os
import sys
import json
import argparse
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

def format_chatml(entry: dict, tokenizer) -> str:
    messages = [
        {"role": "system", "content": "Bạn là chuyên gia phân tích dữ liệu kinh doanh và lập trình viên DuckDB Python cao cấp cho cuộc thi Zalo AI Challenge 2026. Hãy viết mã Python chính xác để trả lời câu hỏi."},
        {"role": "user", "content": entry["prompt"]},
        {"role": "assistant", "content": entry["completion"]}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)

def main():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning with LoRA")
    parser.add_argument("--data_path", type=str, default="train/sft_dataset.jsonl", help="Path to golden SFT jsonl dataset")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Base model ID")
    parser.add_argument("--output_dir", type=str, default="checkpoints/sft_adapter", help="Output directory for LoRA adapter")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Per device train batch size")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit QLoRA")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 (recommended for A100)")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 (for T4)")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"[!] Dataset not found at: {args.data_path}. Please run build_training_dataset.py first.")
        return

    # Load dataset
    print(f"[*] Loading training dataset from: {args.data_path}")
    raw_data = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_data.append(json.loads(line))

    print(f"[+] Loaded {len(raw_data)} golden training samples.")
    if len(raw_data) == 0:
        print("[!] No samples found to train.")
        return

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Format texts
    formatted_texts = [format_chatml(entry, tokenizer) for entry in raw_data]
    dataset = Dataset.from_dict({"text": formatted_texts})

    def tokenize_func(examples):
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized_dataset = dataset.map(tokenize_func, batched=True, remove_columns=["text"])

    # Load Base Model
    print(f"[*] Loading base model: {args.model_id}")
    model_kwargs = {"trust_remote_code": True}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True
        )
    elif args.bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif args.fp16:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map="auto", **model_kwargs)

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="epoch",
        bf16=args.bf16,
        fp16=args.fp16 and not args.bf16,
        gradient_checkpointing=True,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")
    )

    print("[*] Starting SFT Training...")
    trainer.train()

    print(f"[+] Saving final LoRA adapter to: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[+] SFT Training Completed Successfully!")

if __name__ == "__main__":
    main()
