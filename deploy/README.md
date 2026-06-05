# WeDetect ONNX 部署

本目录提供 WeDetect 开放词表检测器的 ONNX 导出与验证工具。整套流程基于
`test_coco_pytorch.py` 中**不依赖 mmcv / mmdet** 的独立 PyTorch 模型实现
（`SimpleYOLOWorldDetector` 视觉塔 + `XLMRobertaLanguageBackbone` 语言塔），
因此 `export_onnx.py` / `eval_onnx.py` 也不再需要 mmdet 运行环境。

WeDetect 是一个**双塔（dual-tower）**开放词表检测器：

```
视觉塔 (vision) : image  ──► ConvNeXt backbone ──► CSPRepBiFPAN neck ──► head(cls_preds / reg_preds)
语言塔 (language): texts  ──► XLM-RoBERTa ──► Linear head ──► L2 normalize ──► txt_feats [K, 768]
融合   (fusion) : BNContrastiveHead   logit = scale · ⟨norm(cls_embed), norm(txt)⟩ + bias
```

两座塔仅在 `BNContrastiveHead` 处汇合。因此导出脚本提供**两种布局**（`--mode`）：

| 模式 | 产物 | 输入 | 输出 |
| --- | --- | --- | --- |
| `whole` | 单个 `*_whole.onnx`（两塔合一） | `image`, `input_ids`, `attention_mask` | `bboxes`, `scores` |
| `dual`  | `*_vision.onnx` + `*_language.onnx`（两塔分离） | vision: `image`, `txt_feats`；language: `input_ids`, `attention_mask` | vision: `bboxes`, `scores`；language: `txt_feats` |

说明：

- **DFL 解码已写入视觉图**：ONNX 输出的 `bboxes` 是已解码的 `xyxy`（位于 letterbox 后的输入坐标系），`scores` 已做 `sigmoid`。宿主代码只需做分数筛选 + NMS + 坐标回缩（见 `eval_onnx.py`）。
- **分词不在 ONNX 内**：`文本 → input_ids/attention_mask` 仍由 HuggingFace tokenizer 在 Python 侧完成，与 `XLMRobertaLanguageBackbone` 行为一致。
- `dual` 模式下，由于检测词表通常固定，**语言塔只需运行一次**得到 `txt_feats`，随后视觉塔逐图复用，推理更高效；`whole` 模式每张图都会重算文本塔。
- 导出与验证脚本均直接复用 `test_coco_pytorch.py` 的模型定义与前/后处理，因此 ONNX 结果可与 `test_coco_pytorch.py` 的 PyTorch 精度**逐项对齐**。

---

## 目录文件

| 文件 | 作用 |
| --- | --- |
| `test_coco_pytorch.py` | 独立（无 mmdet）PyTorch 模型定义 + COCO 评测，是另两个脚本的依赖来源 |
| `export_onnx.py` | 将独立 PyTorch 模型导出为 ONNX（支持 `whole` / `dual`） |
| `eval_onnx.py`   | 在 COCO `val2017` 上验证导出模型的检测精度（mAP，使用 `pycocotools`） |
| `README.md`      | 本说明文档 |

---

## 一、环境

仅需基础 PyTorch / transformers 环境，外加 ONNX 相关依赖（**无需安装 mmcv / mmdet / mmengine**）：

```bash
# 模型 + 分词 + 评测
pip install torch torchvision transformers pillow numpy tqdm pycocotools

# ONNX 导出 + 推理
pip install onnx onnxruntime               # CPU 推理
# 如需 GPU 推理：pip install onnxruntime-gpu （与本机 CUDA 版本匹配）
pip install onnxsim                         # 可选：--simplify 化简计算图
```

需要准备：

- 训练好的权重，例如 `checkpoints/wedetect_base.pth`（mmdet 格式 checkpoint，脚本内部会自动重映射 key）；
- 文本编码器目录，例如 `xlm-roberta-base/`（仓库内已包含 tokenizer 与配置）；
- 评测数据集 COCO `val2017`：标注 `instances_val2017.json` 与图像目录 `val2017/`。

---

## 二、导出 ONNX

### 1. 双塔分离（推荐，`dual`）

```bash
python export_onnx.py \
    --variant base \
    --language-model xlm-roberta-base \
    --checkpoint checkpoints/wedetect_base.pth \
    --mode dual --output-dir onnx_models
# 产物：
#   onnx_models/wedetect_base_vision.onnx
#   onnx_models/wedetect_base_language.onnx
```

### 2. 整体单模型（`whole`）

```bash
python export_onnx.py \
    --variant base \
    --language-model xlm-roberta-base \
    --checkpoint checkpoints/wedetect_base.pth \
    --mode whole --output-dir onnx_models
# 产物：onnx_models/wedetect_base_whole.onnx
```

常用参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--variant` | `base` | 模型规格：`tiny` / `base` / `large` |
| `--language-model` | 必填 | XLM-RoBERTa 目录（如 `xlm-roberta-base`） |
| `--checkpoint` | 必填 | WeDetect 权重 `.pth`（mmdet 格式） |
| `--mode` | `dual` | `whole` 单模型 / `dual` 双塔 |
| `--img-size` | `640` | 视觉塔方形输入尺寸（priors 随之固定，`large` 通常用 `1280`） |
| `--opset` | `17` | ONNX opset |
| `--device` | `cpu` | 加载权重做导出的设备 |
| `--simplify` | 关闭 | 用 `onnxsim` 化简（需安装 `onnxsim`） |

> 提示：导出在 CPU 上即可完成；视觉塔输入固定为 `1×3×img_size×img_size`，文本相关维度（类别数 `num_classes`、序列长度 `seq_len`）为动态轴。

### 张量约定

- 视觉塔输入 `image`：`float32 [1,3,H,W]`，**RGB 且已除以 255**（与 `test_coco_pytorch.py` 的 letterbox 前处理一致）。
- 语言塔输入 `input_ids` / `attention_mask`：`int64 [num_classes, seq_len]`，由 tokenizer 对类别名（中文）批量编码得到。
- 输出 `bboxes`：`float32 [1, N, 4]`，`xyxy`，位于 letterbox 后的输入坐标系（`640` 时 `N = 80²+40²+20² = 8400`）。
- 输出 `scores`：`float32 [1, N, num_classes]`，已 `sigmoid`。

---

## 三、验证精度（COCO val2017）

`eval_onnx.py` 通过 ONNXRuntime 运行网络，并直接复用 `test_coco_pytorch.py` 的
letterbox 前处理、分数筛选 + 类内 NMS + 坐标回缩后处理，以及中文 COCO 词表与
contiguous-id → COCO-id 映射，最终用 `pycocotools` 计算 mAP，因此结果可与
`test_coco_pytorch.py` 的 PyTorch 精度直接对比。

### 1. 双塔模型

```bash
python eval_onnx.py --mode dual \
    --language-model xlm-roberta-base \
    --vision-onnx   onnx_models/wedetect_base_vision.onnx \
    --language-onnx onnx_models/wedetect_base_language.onnx \
    --coco-ann /path/to/instances_val2017.json \
    --coco-img /path/to/val2017
```

### 2. 整体单模型

```bash
python eval_onnx.py --mode whole \
    --language-model xlm-roberta-base \
    --onnx onnx_models/wedetect_base_whole.onnx \
    --coco-ann /path/to/instances_val2017.json \
    --coco-img /path/to/val2017
```

常用参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--mode` | `dual` | 与导出布局一致 |
| `--language-model` | 必填 | 仅用其 tokenizer 对类别名编码 |
| `--coco-ann` | 必填 | `instances_val2017.json` 路径 |
| `--coco-img` | 必填 | `val2017` 图像目录 |
| `--img-size` | `640` | 必须与导出时一致 |
| `--num-images` | `-1` | 仅评测前 N 张（快速冒烟测试），`-1` 为全量 |
| `--device` | `cpu` | `cuda` 需安装 `onnxruntime-gpu` |
| `--score-thr` | `0.001` | 分数阈值（与 PyTorch 评测一致） |
| `--nms-iou` | `0.7` | NMS IoU 阈值 |
| `--pre-nms-topk` / `--post-nms-topk` | `30000` / `300` | NMS 前/后保留框数 |

### 参考精度

ONNX（fp32）结果应与 `test_coco_pytorch.py` 的 PyTorch 精度基本一致（通常 |ΔAP| < 0.2，差异来自算子实现细节）。可先用 `--num-images 100` 做快速校验，再跑全量。

---

## 四、常见问题


- **权重加载提示 missing/unexpected keys**：`load_vision_checkpoint` 以 `strict=False` 加载并会打印重映射结果；只要 backbone / neck / head 的关键权重匹配即可，文本塔权重由 `XLMRobertaLanguageBackbone` 单独加载。
- **ONNXRuntime 找不到算子 / opset 过低**：将 `--opset` 提升至 17 或更高。
- **`whole` 模式评测较慢**：因每张图都会重跑文本塔；离线评测固定词表时建议用 `dual` 模式。
- **类别数不匹配**：ONNX 的 `num_classes` 为动态轴，验证脚本会按内置的 COCO 80 类中文词表自动生成 `txt_feats`，无需重新导出。
- **检测框错位**：请确认 `--img-size` 与导出时一致，且输入未做额外归一化——视觉塔内部不含均值/方差处理，归一化（letterbox 到方形、`/255`）由脚本前处理完成。
