"""
GovConnect Edge - Whisper-Small LoRA Fine-Tuning Pipeline for Google Colab (T4 GPU).
Trained on IndicVoices subset for dialect-adaptive rural speech recognition.

Usage on Google Colab (T4 GPU):
    !pip install -q transformers datasets evaluate jiwer accelerate peft bitsandbytes librosa soundfile
    !python scripts/train_whisper_lora.py --lang ta --output_dir ./models/whisper_lora_ta
"""

import argparse
import os
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper-Small with LoRA on IndicVoices")
    parser.add_argument("--model_name_or_path", type=str, default="openai/whisper-small")
    parser.add_argument("--dataset_name", type=str, default="ai4bharat/indicvoices_r")
    parser.add_argument("--lang", type=str, default="ta", choices=["ta", "hi", "te", "ml"])
    parser.add_argument("--output_dir", type=str, default="./models/whisper_small_lora")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    return parser.parse_args()


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut boss token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def prepare_dataset(batch, feature_extractor, tokenizer):
    audio = batch["audio"]
    # Compute log-Mel input features
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # Encode target text to label ids
    batch["labels"] = tokenizer(batch["transcription"]).input_ids
    return batch


def train(args):
    from datasets import load_dataset, Audio
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )
    import evaluate

    print(f"Loading processor and base model: {args.model_name_or_path} for lang: {args.lang}")
    processor = WhisperProcessor.from_pretrained(
        args.model_name_or_path, language=args.lang, task="transcribe"
    )

    # 8-bit quantized base model for Colab T4 memory efficiency
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        load_in_8bit=True,
        device_map="auto",
    )

    model = prepare_model_for_kbit_training(model)

    # Configure PEFT LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    print(f"Loading dataset: {args.dataset_name} ({args.lang})")
    try:
        raw_dataset = load_dataset(args.dataset_name, args.lang, trust_remote_code=True)
    except Exception as e:
        print(f"Could not load remote dataset directly: {e}. Ensure HF token or local cache.")
        return

    raw_dataset = raw_dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Preprocess
    train_dataset = raw_dataset["train"].map(
        lambda b: prepare_dataset(b, processor.feature_extractor, processor.tokenizer),
        remove_columns=raw_dataset["train"].column_names,
        num_proc=2,
    )
    eval_dataset = raw_dataset["validation"].map(
        lambda b: prepare_dataset(b, processor.feature_extractor, processor.tokenizer),
        remove_columns=raw_dataset["validation"].column_names,
        num_proc=2,
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        evaluation_strategy="epoch",
        fp16=True,
        per_device_eval_batch_size=8,
        generation_max_length=225,
        save_strategy="epoch",
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    print("Starting Whisper-Small LoRA Training on Colab T4...")
    trainer.train()

    # Save trained LoRA adapter weights & processor
    print(f"Saving fine-tuned LoRA weights to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("Training successfully finished!")


if __name__ == "__main__":
    args = parse_args()
    train(args)
