# WeDetect: 以检索来实现快速开放词汇目标检测

## 👀 WeDetect家族介绍

<p align="left">
    <img src="./assets/model.png" width="800px">
</p>


开放词汇检测旨在利用文本描述来检测任意的物体，现有的不利用跨模态交互的方案把识别任务类比成一种检索任务，即在一个统一的特征空间中匹配区域特征和文本特征。在这个项目中，我们进一步探索了这种检索的思想，并展示其在高效性和多功能性方面的优势，并提出了一个模型家族WeDetect：
- **领先的开放词汇性能。** **WeDetect** 是一个实时监测器，其具有双塔结构。我们发现，在精心收集的数据和充分的训练下，没有使用跨模态交互的WeDetect模型的性能超过了其他采用融合层的模型，使得WeDetect成为一个超强的检测基础模型。
- **高效的历史数据检索** **WeDetect-Uni** 是一个基于WeDetect的通用区域候选框生成器。我们冻结整个检测器，仅对目标性提示（prompt）进行微调。重要的是，候选框的特征是类别有关的，因此能够支持一种新的应用，即**目标检索**，支持在历史数据中检索目标。
- **与多模态大模型结合来完成复杂指代理解任务 (REC).** **WeDetect-Ref**是一个基于多模态大模型的物体分类模型，它能在给定的候选框中，检索出与给定文本相关的框。该模型不再采用逐词预测（next-token prediction）的形式，能够在一次模型推理中完成对所有物体的分类，因此具有极高的推理效率。

综上，WeDetect模型家族统一了目标检测、候选框生成、物体检索、指代表达式理解等区域感知与理解任务，并在15个公开数据集上取得领先性能。

## 📈 实验结果

#### 📍 模型库

- 请将下列模型下载并放置在 `checkpoints`文件夹中.
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

#### 📍 结果

<p align="left">
    <img src="./assets/performance1.png" width="800px">
</p>
<p align="left">
    <img src="./assets/performance2.png" width="800px">
</p>


## 🔧 安装环境

#### 基本库

```
pytorch==2.5.1+cu124
transformers==4.57.1
trl==0.17.0
accelerate==1.10.0
mmcv==2.1.0
mmdet==3.3.0
mmengine==0.10.7
```

- 请按照下列的指令安装环境

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

- 请注意：WeDetect是一个中文模型，请传入中文类名。同时该模型支持同时检测多个类别，每个类名用**英文**逗号隔开，命令中所有字符均为英文字符，包括引号（除了中文的类名）。


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

- WeDetect-Ref是一个多语言模型，您可以使用中文或者英文表达式进行测试，但是每次仅能传入一个表达式。



### 📏 评测
#### 📍 WeDetect
```
# Evaluating WeDetect-Base on COCO
bash dist_test.sh config/wedetect_base.py /PATH/TO/WEDETECT 8
```
- 请您修改config中的数据路径

#### 📍 WeDetect-Uni
```
# Evaluating recall on COCO
cd eval_recall
torchrun --nproc-per-node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=29500 eval_recall.py --wedetect_uni_checkpoint wedetect_base_uni.pth --dataset coco
```
- 请您修改`eval_recall/eval_recall.py`中第1400的数据路径
- Dataset的选项可为`coco`，`lvis`和`paco`

```
# Evaluating the object retrieval task on COCO

cd eval_retrieval

# extract embedding
torchrun --nproc-per-node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=29500 extract_embedding.py --model wedetect --wedetect_checkpoint wedetect_base.pth --wedetect_uni_checkpoint wedetect_base_uni.pth --dataset coco

# retrieval
python3 retrieval_metric.py --model wedetect --dataset coco --thre 0.2
```
- 请您修改`eval_retrieval/extract_embedding.py`中第1323行的数据路径、`eval_retrieval/retrieval_metric.py`中第61行的数据路径、以及`eval_retrieval/retrieval_metric.py`中第82行的数据路径
- Dataset可以是`coco`和`lvis`

#### 📍 WeDetect-Ref
- 请您参见`wedetect_ref`文件夹


### 🙏 致谢

- 本项目基于[mmdetection](https://github.com/open-mmlab/mmdetection/)、[YOLO-World](https://github.com/AILab-CVC/YOLO-World)、[transformers](https://github.com/huggingface/transformers)、[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) 等项目开发，感谢这些优秀的开源项目。

### ✒️ 引用

如果您觉得我们的工作对您的研究有帮助，请您引用我们的工作：   

```bibtex
@article{fu2025wedetect,
  title={WeDetect: Fast Open-Vocabulary Object Detection as Retrieval},
  author={Fu, Shenghao and Su, Yukun and Rao, Fengyun and LYU, Jing and Xie, Xiaohua and Zheng, Wei-Shi},
  journal={arXiv preprint arXiv:2512.12309},
  year={2025}
}
```

## 📜 协议

- 我们的模型和代码在GPL-v3协议下开源。

