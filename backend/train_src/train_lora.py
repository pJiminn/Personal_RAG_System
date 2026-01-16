# train_src/train_lora.py
import argparse
import os
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to dataset")
    parser.add_argument("--base_model", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    args = parser.parse_args()

    # 1) 데이터 로드 (Azure ML에서 Input으로 제공됨)
    raw_dataset = load_dataset("json", data_files=args.data_path)["train"]

    # 2) 모델/토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # 3) LoRA 설정
    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # 4) 토큰화 및 학습 (기존 로직 유지)
    def format_example(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        tokenized = tokenizer(text, truncation=True, max_length=1024, padding="max_length")
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = raw_dataset.map(format_example)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        fp16=True,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)
    trainer.train()

    # 5) 모델 저장 (Azure ML Job의 output 경로)
    model.save_pretrained(args.output_dir)

if __name__ == "__main__":
    train()