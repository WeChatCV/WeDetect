# Copyright (c) Tencent Inc. All rights reserved.
"""WeDetect-Anything Gradio Demo.

A thin UI wrapper around the WeDetect-Anything pipeline:
    proposal ONNX (objectness) -> class-agnostic NMS -> box-embedding / text
    matching against ``prompt_free.npy`` -> draw Chinese labels.

The heavy lifting is shared with ``infer_anything.py`` so the demo and the CLI
stay perfectly consistent.

Run:
    python wedetect_anything/wedetect_unified_app.py
"""
import os
import os.path as osp
import sys

import numpy as np
import onnxruntime
from PIL import Image, ImageDraw, ImageFont

import gradio as gr

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from infer_anything import (  # noqa: E402
    build_level_constants,
    letterbox,
    load_meta,
    np_sigmoid,
    proposal_nms,
    rescale,
)
from prompt_free_vocab import prompt_free_name  # noqa: E402

# ============================================================================
#                         配置（按需修改）
# ============================================================================
ONNX_PATH = os.environ.get(
    "WEDETECT_ONNX", "onnx_models/wedetect_anything_base.onnx")
META_PATH = os.environ.get(
    "WEDETECT_META", "onnx_models/wedetect_anything_base_meta.json")
TEXT_FEAT_PATH = os.environ.get("WEDETECT_TEXT_FEAT", "prompt_free.npy")
FONT_PATH = os.environ.get("WEDETECT_FONT", "../simsun.ttc")
IMG_SIZE = int(os.environ.get("WEDETECT_IMG_SIZE", "640"))

VOCAB = list(prompt_free_name)
print(f"[init] 词表大小: {len(VOCAB)}")

print(f"[init] 加载 ONNX: {ONNX_PATH}")
SESSION = onnxruntime.InferenceSession(
    ONNX_PATH, providers=['CPUExecutionProvider'])
META = load_meta(META_PATH)
BOX_SCALES, BOX_BIASES = build_level_constants(META)
TEXT_FEAT = np.load(TEXT_FEAT_PATH).astype(np.float32)
print(f"[init] 文本特征: {TEXT_FEAT.shape}")

try:
    CHINESE_FONT = ImageFont.truetype(FONT_PATH, 24)
except Exception as e:
    print(f"⚠️ 中文字体加载失败，使用默认字体: {e}")
    CHINESE_FONT = ImageFont.load_default()


# ============================================================================
#                         检测主逻辑
# ============================================================================
def run_detection(input_img, match_thr=0.4, score_thr=0.1,
                  iou_thr=0.7, keep_top_k=100):
    if input_img is None:
        return None, "请先上传一张图片"

    origin = np.asarray(input_img.convert("RGB"))     # RGB
    ori_h, ori_w = origin.shape[:2]
    img_lb, _, _ = letterbox(origin.copy(), (IMG_SIZE, IMG_SIZE))
    inp = (img_lb / 255.0).transpose(2, 0, 1)[None].astype(np.float32)
    inp = np.ascontiguousarray(inp)

    scores, bboxes, embeds = SESSION.run(
        ['scores', 'bboxes', 'embeds'], {'input_image': inp})
    scores, bboxes, embeds = scores[0], bboxes[0], embeds[0]

    keep_idxs, _ = proposal_nms(
        scores, bboxes, float(score_thr), float(iou_thr), int(keep_top_k))
    img_show = input_img.convert("RGB").copy()
    if keep_idxs.numel() == 0:
        return img_show, "未检测到任何目标"

    kept_boxes = bboxes[keep_idxs.numpy()]
    kept_embeds = embeds[keep_idxs.numpy()]
    kept_scales = BOX_SCALES[keep_idxs.numpy()]
    kept_biases = BOX_BIASES[keep_idxs.numpy()]

    draw = ImageDraw.Draw(img_show)
    rng = np.random.default_rng(0)
    detected = []
    for i in range(len(keep_idxs)):
        logit = (TEXT_FEAT @ kept_embeds[i]) * kept_scales[i] + kept_biases[i]
        prob = np_sigmoid(logit)
        cls = int(np.argmax(prob))
        conf = float(prob[cls])
        if conf < float(match_thr):
            continue
        box = rescale((IMG_SIZE, IMG_SIZE), kept_boxes[i].copy(),
                      (ori_h, ori_w)).round()
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        label = f"{VOCAB[cls]} {conf:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        tw = max(len(label) * 16, 60)
        draw.rectangle([x1, y1, x1 + tw, y1 + 26], fill=color)
        draw.text((x1 + 3, y1 + 1), label, font=CHINESE_FONT, fill="white")
        detected.append(f"{VOCAB[cls]}（{conf:.3f}）")

    info = (f"✅ 候选框 {len(keep_idxs)} 个，过滤后输出 {len(detected)} 个目标\n"
            f"📋 标签：{', '.join(detected) if detected else '无'}")
    return img_show, info


# ============================================================================
#                            Gradio 界面
# ============================================================================
CUSTOM_CSS = """
.gradio-container { max-width: 100% !important; padding: 10px 20px !important; }
.input-section { padding: 15px; background-color: #f0f2f6; border-radius: 8px; margin-bottom: 10px; }
"""

with gr.Blocks(title="WeDetect 万物检测 Demo", theme=gr.themes.Soft(),
               css=CUSTOM_CSS) as demo:
    gr.Markdown("# 🌐 WeDetect 万物检测 Demo")
    gr.Markdown("基于 WeDetect-Uni 提取类别无关候选框，再用每框 embedding 与"
                "超大词表文本特征做相似度匹配，模拟万物检测。")

    with gr.Row():
        with gr.Column(scale=4, elem_classes=["input-section"]):
            detect_input = gr.Image(type="pil", label="上传待检测图片",
                                    height=380, image_mode="RGB", container=True)

            example_dir = "./examples"
            examples = [[osp.join(example_dir, f)]
                        for f in ("1.jpg", "2.jpg", "3.png")
                        if osp.exists(osp.join(example_dir, f))]
            if examples:
                gr.Examples(examples=examples, inputs=[detect_input],
                            label="📸 候选示例图片", examples_per_page=5)

            with gr.Row():
                match_thr = gr.Slider(0.1, 0.9, value=0.4, step=0.05,
                                      label="文本匹配阈值（match_thr）")
                score_thr = gr.Slider(0.05, 0.9, value=0.1, step=0.05,
                                      label="objectness 阈值")
            with gr.Row():
                iou_thr = gr.Slider(0.3, 0.95, value=0.7, step=0.05,
                                    label="NMS IoU 阈值")
                keep_top_k = gr.Slider(10, 300, value=100, step=10,
                                       label="保留 Top-K 候选框")
            detect_btn = gr.Button("🚀 开始检测", variant="primary", size="lg")

        with gr.Column(scale=6):
            detect_output_image = gr.Image(label="检测结果", type="pil",
                                           interactive=False, height=600,
                                           container=True)
            detect_output_info = gr.Markdown("*等待检测...*")

    detect_btn.click(
        fn=run_detection,
        inputs=[detect_input, match_thr, score_thr, iou_thr, keep_top_k],
        outputs=[detect_output_image, detect_output_info])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8081, share=True,
                debug=True, allowed_paths=[os.getcwd(), "./examples"])
