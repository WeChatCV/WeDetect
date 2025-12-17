# WeDetect: Fast Open-Vocabulary Object Detection as Retrieval

This is the official PyTorch implementation of WeDetect. Our paper can be found at [here](https://arxiv.org/abs/2512.12309).

If you find our work helpful, please kindly give us a star 🌟

Here is the [中文版指南](./README_zh.md).

## 👀 WeDetect Family Overview

<p align="left">
    <img src="./assets/model.png" width="800px">
</p>

Open-vocabulary object detection aims to detect arbitrary classes via text prompts. Methods without cross-modal fusion layers (non-fusion) offer faster inference by treating recognition as a retrieval problem, \ie, matching regions to text queries in a shared embedding space. In this work, we fully explore this retrieval philosophy and demonstrate its unique advantages in efficiency and versatility through a model family named WeDetect: 
- **State-of-the-art performance.** **WeDetect** is a real-time detector with a dual-tower architecture. We show that, with well-curated data and full training, the non-fusion WeDetect surpasses other fusion models and establishes a strong open-vocabulary foundation. 
- **Fast backtrack of historical data.** **WeDetect-Uni** is a universal proposal generator based on WeDetect. We freeze the entire detector and only finetune an objectness prompt to retrieve generic object proposals across categories. Importantly, the proposal embeddings are class-specific and enable a new application, **object retrieval**, supporting retrieval objects in historical data.
- **Integration with LMMs for referring expression comprehension (REC).** We further propose **WeDetect-Ref**, an LMM-based object classifier to handle complex referring expressions, which retrieves target objects from the proposal list extracted by WeDetect-Uni. It discards next-token prediction and classifies objects in a single forward pass. 

Together, the WeDetect family unifies detection, proposal generation, object retrieval, and REC under a coherent retrieval framework, achieving state-of-the-art performance across 15 benchmarks with high inference efficiency.

## 📈 Experimental Results

#### 📍 Model Zoo

- Please download the models and put them in `checkpoints`.
- WeDetect
  - [WeDetect-Tiny](https://huggingface.co/fushh7/WeDetect)
  - [WeDetect-Base](https://huggingface.co/fushh7/WeDetect)
  - [WeDetect-Large](https://huggingface.co/fushh7/WeDetect)

- WeDetect-Uni
  - [WeDetect-Base-Uni](https://huggingface.co/fushh7/WeDetect)
  - [WeDetect-Large-Uni](https://huggingface.co/fushh7/WeDetect)

- WeDetect-Ref
  - [WeDetect-Ref 2B](https://huggingface.co/fushh7/WeDetect-Ref-2B)
  - [WeDetect-Ref 4B](https://huggingface.co/fushh7/WeDetect-Ref-4B)

#### 📍 Results

<p align="left">
    <img src="./assets/performance1.png" width="800px">
</p>
<p align="left">
    <img src="./assets/performance2.png" width="800px">
</p>

## 🔧 Install

#### Our environment

```
pytorch==2.5.1+cu124
transformers==4.57.1
trl==0.17.0
accelerate==1.10.0
mmcv==2.1.0
mmdet==3.3.0
mmengine==0.10.7
```

- MMCV series packages are not required for WeDetect-Ref users.
- Install the environment as follows.

```
pip install transformers==4.57.1 trl==0.17.0 accelerate==1.10.0 -i https://mirrors.cloud.tencent.com/pypi/simple
pip install pycocotools terminaltables jsonlines tabulate lvis supervision==0.19.0 webdataset ddd-dataset -i https://mirrors.cloud.tencent.com/pypi/simple

pip install openmim -i https://mirrors.cloud.tencent.com/pypi/simple
mim install mmcv==2.1.0
mim install mmdet==3.3.0
```



## ⭐ Demo

#### 📍 WeDetect
```
python3 infer_wedetect.py --config config/wedetect_large.py --checkpoint checkpoints/wedetect_large.pth --image assets/demo.jpeg --text '鞋,床' --threshold 0.3
```
<p align="left">
    <img src="./assets/pred_wedetect_large.jpeg" width="800px">
</p>

- Note: WeDetect is a Chinese-language model, so please provide class names in Chinese. The model supports detecting multiple categories simultaneously by separating each class name with an English comma. All characters in the command should be in English, including quotation marks (except for the Chinese class names).



#### 📍 WeDetect-Uni

```
# output the prediction higher than the threshold
python generate_proposal.py --wedetect_uni_checkpoint /PATH/TO/WEDETECT_UNI --image assets/demo.jpeg --visualize --score_thre 0.2
```
<p align="left">
    <img src="./assets/pred_wedetect_uni_large.png" width="800px">
</p>

#### 📍 WeDetect-Ref
```
# output the top1 prediction
python infer_wedetect_ref.py --wedetect_ref_checkpoint /PATH/TO/WEDETECT_REF --wedetect_uni_checkpoint /PATH/TO/WEDETECT_UNI --image assets/demo.jpeg --query "a photo of trees and a river" --visualize

# output the prediction higher than the threshold
python infer_wedetect_ref.py --wedetect_ref_checkpoint /PATH/TO/WEDETECT_REF --wedetect_uni_checkpoint /PATH/TO/WEDETECT_UNI --image assets/demo.jpeg --query "a photo of trees and a river" --visualize --score_thre 0.3
```
<p align="left">
    <img src="./assets/pred_wedetect_ref_4b.png" width="800px">
</p>

- WeDetect-Ref is a multilingual model. You can use either Chinese or English queries for testing, but only one query can be provided at a time.


### 📏 Evaluation
#### 📍 WeDetect
```
# Evaluating WeDetect-Base on COCO
bash dist_test.sh config/wedetect_base.py /PATH/TO/WEDETECT 8
```
- Please change the dataset path in the config.

#### 📍 WeDetect-Uni
```
# Evaluating recall on COCO
cd eval_recall
torchrun --nproc-per-node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=29500 eval_recall.py --wedetect_uni_checkpoint wedetect_base_uni.pth --dataset coco
```
- Please change the dataset path in Line 10 of `eval_recall/eval_recall.py`.
- Dataset can be `coco`, `lvis`, and `paco`.

```
# Evaluating the object retrieval task on COCO

cd eval_retrieval

# extract embedding
torchrun --nproc-per-node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=29500 extract_embedding.py --model wedetect --wedetect_checkpoint wedetect_base.pth --wedetect_uni_checkpoint wedetect_base_uni.pth --dataset coco

# retrieval
python3 retrieval_metric.py --model wedetect --dataset coco --thre 0.2
```
- Please change the dataset path in Line 1323 of `eval_retrieval/extract_embedding.py` and Line 61 of `eval_retrieval/retrieval_metric.py` and Line 82 of `eval_retrieval/retrieval_metric.py`.
- Dataset can be `coco`, and `lvis`.

#### 📍 WeDetect-Ref
- Please refer to the folder `wedetect_ref`.

### 🙏 Acknowledgement

- WeDetect is based on many outstanding open-sourced projects, including [mmdetection](https://github.com/open-mmlab/mmdetection/), [YOLO-World](https://github.com/AILab-CVC/YOLO-World), [transformers](https://github.com/huggingface/transformers), [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) and many others. Thank the authors of above projects for open-sourcing their assets!

### ✒️ Citation

If you find our work helpful for your research, please consider citing our work.   

```bibtex
@article{fu2025wedetect,
  title={WeDetect: Fast Open-Vocabulary Object Detection as Retrieval},
  author={Fu, Shenghao and Su, Yukun and Rao, Fengyun and LYU, Jing and Xie, Xiaohua and Zheng, Wei-Shi},
  journal={arXiv preprint arXiv:2512.12309},
  year={2025}
}
```

## 📜 License

- Our models and code are under the GPL-v3 Licence.

