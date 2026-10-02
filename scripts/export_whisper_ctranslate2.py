"""
GovConnect Edge - Convert fine-tuned Whisper-Small model to CTranslate2 INT8 format
for ultra-fast edge inference under Pi-profile constraints.

Usage:
    python scripts/export_whisper_ctranslate2.py --model_dir ./models/whisper_small_lora --output_dir ./models/whisper_ct2_int8
"""

import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Whisper model to CTranslate2 INT8")
    parser.add_argument("--model_dir", type=str, default="./models/whisper_small_lora", help="HuggingFace model dir")
    parser.add_argument("--output_dir", type=str, default="./models/whisper_ct2_int8", help="Target CT2 directory")
    parser.add_argument("--quantization", type=str, default="int8", choices=["int8", "int8_float16", "float16", "int16"])
    return parser.parse_args()


def export(args):
    print(f"Exporting Whisper model from {args.model_dir} to CTranslate2 ({args.quantization})...")
    cmd = [
        "ct2-transformers-converter",
        "--model", args.model_dir,
        "--output_dir", args.output_dir,
        "--quantization", args.quantization,
        "--low_cpu_mem_usage",
        "--force",
    ]
    try:
        subprocess.check_call(cmd)
        print(f"Successfully converted model to CTranslate2 INT8 format at: {args.output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed: {e}")
        print("Ensure 'ctranslate2' is installed: pip install ctranslate2")
        sys.exit(1)


if __name__ == "__main__":
    args = parse_args()
    export(args)
