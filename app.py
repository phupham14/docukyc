"""Simple Streamlit UI for the Qwen2.5-VL document assistant."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any

import streamlit as st
import torch
from PIL import Image


BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct")
ADAPTER_PATH = Path(
    os.getenv("LORA_PATH", str(Path(__file__).parent / "qwen2.5_vl_lora_final"))
)
# Model OpenVINO tạo bởi export_openvino.py; nếu có sẽ được ưu tiên khi không có GPU.
OV_MODEL_PATH = Path(
    os.getenv("OV_MODEL_PATH", str(Path(__file__).parent / "qwen2.5_vl_ov_int8"))
)
# Chiều cao (px) của khung ảnh; cột bên phải được canh cho bằng khung này.
PANEL_HEIGHT = 620

# CPU lai (P-core + E-core): dùng hết 12 luồng làm decode chậm hơn ~4 lần so với 8
# luồng (đo trên Core Ultra 7 255U). Đặt ở top-level vì số luồng OpenMP gắn với
# từng thread, mà Streamlit chạy mỗi lần rerun trên một thread mới.
if not torch.cuda.is_available():
    torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "8")))


def _device_settings() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return "auto", dtype
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        # GPU Intel (Arc / Core Ultra) khi cài PyTorch bản XPU.
        return "xpu", torch.float16
    return "cpu", torch.float32


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.synchronize()


def _openvino_device() -> str:
    """GPU Intel tích hợp nhanh hơn CPU ~5 lần ở bước đọc ảnh (đo trên Core Ultra 7 255U)."""
    device = os.getenv("OV_DEVICE")
    if device:
        return device
    import openvino as ov

    return "GPU" if "GPU" in ov.Core().available_devices else "CPU"


def _load_openvino() -> tuple[Any, Any]:
    from optimum.intel import OVModelForVisualCausalLM
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(OV_MODEL_PATH))
    model = OVModelForVisualCausalLM.from_pretrained(
        str(OV_MODEL_PATH),
        device=_openvino_device(),
        # Cache model đã compile để các lần khởi động sau nạp nhanh hơn.
        ov_config={"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": str(OV_MODEL_PATH / "cache")},
    )
    return model, processor


@st.cache_resource(show_spinner=False)
def load_model() -> tuple[Any, Any, str]:
    """Load the model once per app process.

    Returns (model, processor, backend). Uses the OpenVINO export when it exists
    and there is no CUDA GPU; otherwise base model + LoRA in PyTorch.
    """
    if not torch.cuda.is_available() and (OV_MODEL_PATH / "openvino_config.json").is_file():
        model, processor = _load_openvino()
        return model, processor, f"OpenVINO {model._device}"

    if not ADAPTER_PATH.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy LoRA adapter tại '{ADAPTER_PATH}'. "
            "Hãy đặt thư mục qwen2.5_vl_lora_final cạnh app.py "
            "hoặc cấu hình biến môi trường LORA_PATH."
        )

    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    device_map, dtype = _device_settings()
    processor = AutoProcessor.from_pretrained(str(ADAPTER_PATH))
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=dtype,
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))
    # Gộp LoRA vào trọng số gốc để generate không phải tính thêm nhánh adapter.
    model = model.merge_and_unload()
    model.eval()
    return model, processor, "PyTorch"


def evaluate_document(
    model: Any,
    processor: Any,
    image_path: str,
    question: str,
    max_new_tokens: int,
    max_pixels: int,
) -> tuple[str, dict[str, float]]:
    """Run the same image-question inference flow used in the notebook.

    Returns the decoded answer and timing stats (token counts, prefill time,
    decode speed).
    """
    from qwen_vl_utils import process_vision_info

    # Không truyền system message (giống notebook) để chat template dùng mặc định
    # "You are a helpful assistant.". Mọi mẫu fine-tune đều có system prompt
    # "document intelligence assistant" đi kèm đáp án JSON, nên gửi lại prompt đó
    # làm model trả JSON cả với câu hỏi thường.
    messages = [
        {
            "role": "user",
            "content": [
                # max_pixels giới hạn số token ảnh (mỗi token ~ 28x28 pixel).
                {"type": "image", "image": image_path, "max_pixels": max_pixels},
                {"type": "text", "text": question},
            ],
        },
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    model_device = model.device
    inputs = {
        key: value.to(model_device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }

    from transformers import StoppingCriteria, StoppingCriteriaList

    class FirstTokenTimer(StoppingCriteria):
        """Ghi lại thời điểm sinh xong token đầu tiên (kết thúc prefill)."""

        def __init__(self) -> None:
            self.first_token_at: float | None = None

        def __call__(self, input_ids, scores, **kwargs):
            if self.first_token_at is None:
                _synchronize()
                self.first_token_at = time.perf_counter()
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    timer = FirstTokenTimer()
    generate_start = time.perf_counter()
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            stopping_criteria=StoppingCriteriaList([timer]),
        )
    _synchronize()
    generate_end = time.perf_counter()

    input_length = inputs["input_ids"].shape[1]
    generated_ids = generated_ids[:, input_length:]
    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    output_tokens = generated_ids.shape[1]
    first_token_at = timer.first_token_at or generate_end
    decode_seconds = generate_end - first_token_at
    return text, {
        "input_tokens": input_length,
        "output_tokens": output_tokens,
        "prefill_seconds": first_token_at - generate_start,
        "decode_tok_per_s": (output_tokens - 1) / decode_seconds if decode_seconds > 0 else 0.0,
    }


def log_model_info(model: Any, backend: str, load_seconds: float) -> None:
    """In thông tin backend / thiết bị và thời gian nạp model ra terminal."""
    device = model.device
    if backend == "PyTorch":
        info = f"Backend: PyTorch | Thiết bị: {device}"
        if device.type == "cpu":
            info += f" | Số luồng CPU: {torch.get_num_threads()}"
    else:
        info = f"Backend: {backend}"
    print(f"[DocuKYC] {info} | Thời gian nạp model: {load_seconds:.1f}s", flush=True)
    if backend == "PyTorch" and device.type == "cpu":
        print(
            "[DocuKYC] Cảnh báo: đang chạy PyTorch trên CPU, inference sẽ rất chậm. "
            "Chạy `python export_openvino.py` để chuyển sang OpenVINO INT8.",
            flush=True,
        )


def log_stats(stats: dict[str, float], infer_seconds: float, total_seconds: float) -> None:
    """In thông số thời gian ra terminal thay vì hiển thị trên giao diện."""
    print(
        f"[DocuKYC] Tổng: {total_seconds:.1f}s (inference {infer_seconds:.1f}s) | "
        f"Đọc ảnh + prompt: {stats['prefill_seconds']:.1f}s | "
        f"Token vào/ra: {stats['input_tokens']}/{stats['output_tokens']} | "
        f"Tốc độ sinh: {stats['decode_tok_per_s']:.1f} tok/s",
        flush=True,
    )


def main() -> None:
    st.set_page_config(page_title="DocuKYC", page_icon="📄", layout="wide")
    st.title("📄 DocuKYC")
    st.caption("Hỏi đáp và trích xuất thông tin từ ảnh tài liệu")

    # Nạp model ngay khi mở trang (chỉ chậm ở lần đầu của mỗi process Streamlit),
    # các lần rerun / bấm "Phân tích" sau sẽ lấy từ cache.
    try:
        load_start = time.perf_counter()
        with st.spinner("Đang nạp model (chỉ lần đầu)..."):
            model, processor, backend = load_model()
        load_seconds = time.perf_counter() - load_start
    except Exception as exc:
        st.error(f"Không thể nạp model: {exc}")
        st.info(
            "Kiểm tra dependencies và đặt thư mục qwen2.5_vl_lora_final "
            "cạnh app.py trước khi thử lại."
        )
        return

    # Chỉ in khi model thực sự được nạp; các lần rerun lấy từ cache gần như tức thì.
    if load_seconds > 1:
        log_model_info(model, backend, load_seconds)

    uploaded_file = st.file_uploader(
        "Tải ảnh tài liệu lên",
        type=["png", "jpg", "jpeg", "webp"],
        help="Ảnh sẽ chỉ được lưu tạm trong lúc chạy inference.",
    )

    if uploaded_file is None:
        st.info("Hãy tải lên một ảnh tài liệu để bắt đầu.")
        return

    image = Image.open(uploaded_file).convert("RGB")
    preview_col, input_col = st.columns([1, 1])
    with preview_col:
        # Giới hạn chiều cao để ảnh dọc không đẩy cả trang dài ra; ảnh cuộn bên trong khung.
        with st.container(height=PANEL_HEIGHT, border=True):
            st.image(image, use_container_width=True)

    with input_col:
        question = st.text_area(
            "Câu hỏi",
            value="What is the primary subject or purpose of this document?",
            height=100,
        )
        button_col, settings_col = st.columns([5, 1])
        analyze = button_col.button("🚀 Phân tích", type="primary", use_container_width=True)
        with settings_col.popover("⚙️", use_container_width=True, help="Thiết lập"):
            max_tokens = st.slider("Số token trả về tối đa", 64, 2048, 512, 64)
            max_image_tokens = st.select_slider(
                "Số token ảnh tối đa",
                options=[256, 384, 512, 768, 1024, 1280, 1600, 2048, 4096],
                value=768,
                help=(
                    "Ảnh được resize để không vượt quá số token này. Càng nhỏ càng "
                    "nhanh, nhưng chữ nhỏ có thể bị đọc sai."
                ),
            )

        st.markdown("**Kết quả**")
        # Kết quả nằm ngay dưới nút bấm, cùng cột với câu hỏi nên không phải cuộn trang.
        result_box = st.container(height=PANEL_HEIGHT - 230, border=True)

    if not analyze:
        result_box.caption("Bấm “Phân tích” để xem câu trả lời.")
        return
    if not question.strip():
        result_box.warning("Vui lòng nhập câu hỏi.")
        return

    start = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        image.save(temp_file.name, format="PNG")
        temp_path = temp_file.name

    try:
        with result_box, st.spinner("Đang phân tích tài liệu..."):
            infer_start = time.perf_counter()
            result, stats = evaluate_document(
                model,
                processor,
                temp_path,
                question.strip(),
                max_tokens,
                max_image_tokens * 28 * 28,
            )
            _synchronize()
            infer_seconds = time.perf_counter() - infer_start
    except Exception as exc:
        result_box.error(f"Không thể phân tích tài liệu: {exc}")
        return
    finally:
        Path(temp_path).unlink(missing_ok=True)
    total_seconds = time.perf_counter() - start

    log_stats(stats, infer_seconds, total_seconds)
    result_box.write(result)


if __name__ == "__main__":
    main()
