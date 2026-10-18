"""Gộp LoRA vào Qwen2.5-VL rồi export sang OpenVINO (weight INT8) để chạy nhanh trên CPU Intel.

Chạy một lần (nên tắt app Streamlit trước để đủ RAM):
    python export_openvino.py

Kết quả nằm ở thư mục qwen2.5_vl_ov_int8 cạnh file này; app.py sẽ tự dùng nếu thấy.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent


def merge_lora(base_model_id: str, adapter_path: Path, merged_path: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print(f"[1/2] Gộp LoRA '{adapter_path}' vào '{base_model_id}'...", flush=True)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_id, dtype=torch.bfloat16, device_map="cpu"
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path)).merge_and_unload()
    model.save_pretrained(str(merged_path), safe_serialization=True)
    # Processor/tokenizer lấy từ thư mục adapter vì đó là bản dùng lúc fine-tune.
    AutoProcessor.from_pretrained(str(adapter_path)).save_pretrained(str(merged_path))


def export_openvino(merged_path: Path, output_path: Path, weight_format: str) -> None:
    print(f"[2/2] Export OpenVINO ({weight_format}) -> '{output_path}'...", flush=True)
    cli = shutil.which("optimum-cli")
    command = [cli] if cli else [sys.executable, "-m", "optimum.commands.optimum_cli"]
    command += [
        "export", "openvino",
        "--model", str(merged_path),
        "--task", "image-text-to-text",
        "--weight-format", weight_format,
        str(output_path),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    # Console Windows (cp125x) không in được tiếng Việt khi stdout bị redirect.
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter", type=Path, default=HERE / "qwen2.5_vl_lora_final")
    parser.add_argument("--merged", type=Path, default=HERE / "qwen2.5_vl_merged")
    parser.add_argument("--output", type=Path, default=HERE / "qwen2.5_vl_ov_int8")
    parser.add_argument("--weight-format", default="int8", choices=["int8", "int4", "fp16"])
    parser.add_argument(
        "--keep-merged", action="store_true", help="Giữ lại bản merged (~7.5GB) sau khi export."
    )
    args = parser.parse_args()

    start = time.perf_counter()
    if not (args.merged / "config.json").is_file():
        merge_lora(args.base_model, args.adapter, args.merged)
    export_openvino(args.merged, args.output, args.weight_format)
    if not args.keep_merged:
        shutil.rmtree(args.merged, ignore_errors=True)
    print(f"Xong sau {time.perf_counter() - start:.0f}s: {args.output}", flush=True)


if __name__ == "__main__":
    main()
