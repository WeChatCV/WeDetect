# Copyright (c) Tencent Inc. All rights reserved.
import os
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont
import datetime
import json
def visualize_batch(batch_inputs, batch_data_samples, output_dir="debug_vis"):
    """
    可视化训练批次的图像和边界框
    
    参数:
        batch_inputs (Tensor): 输入的图像张量 (B, C, H, W)
        batch_data_samples (dict): 包含边界框和文本的数据结构
        output_dir (str): 输出图像的保存目录
    """
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用时间戳确保不同批次不会混淆
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    batch_dir = os.path.join(output_dir, f"batch_{timestamp}")
    os.makedirs(batch_dir, exist_ok=True)
    
    # # 反归一化参数 (ImageNet)
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])

    # # # 反归一化参数 (WeCLip)
    # mean=[0.48145466, 0.4578275, 0.40821073]
    # std=[0.26862954, 0.26130258, 0.27577711]

    # 反归一化参数
    mean = np.array([0, 0, 0])
    std = np.array([1, 1, 1])
    
    # 加载中文字体
    chinese_font = None
    try:
        font_path = "/mnt/csp/sh1/home/yukunsu/code/simsun.ttc"
        chinese_font = ImageFont.truetype(font_path, 14)
    except:
        print("警告：无法加载中文字体，将使用默认字体")
    
    # 转换Tensor到NumPy并反归一化
    images = batch_inputs.detach().cpu().numpy()
    images = images.transpose(0, 2, 3, 1)  # (B, H, W, C)
    images = images * std + mean  # 反归一化
    images = np.clip(images * 255, 0, 255).astype(np.uint8)
    
    # 解析边界框数据
    bboxes_labels = batch_data_samples["bboxes_labels"].cpu().numpy()
    texts_list = batch_data_samples["texts"]

    # print('batch_data_samples', batch_data_samples)
    
    # 遍历批处理中的每个图像
    for img_idx in range(len(images)):
        # 转换为PIL图像
        img = images[img_idx].copy()
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        
        # 提取当前图像的边界框
        img_bboxes = bboxes_labels[bboxes_labels[:, 0] == img_idx]
        
        # 绘制边界框和文本
        for bbox in img_bboxes:
            # 解析边界框数据 [img_idx, class_idx, x1, y1, x2, y2]
            class_idx = int(bbox[1])
            # x1, y1, w,h = map(int, bbox[2:6])
            # x2,y2=x1+w,y1+h
            x1, y1, x2,y2 = map(int, bbox[2:6])
            
            # 获取文本标签
            if class_idx < len(texts_list[img_idx]):
                label = texts_list[img_idx][class_idx].strip()
                if len(label) > 15:  # 截断长文本
                    label = label[:12] + "..."
            else:
                label = f"Class {class_idx}"
            
            # 跳过空文本
            if not label:
                label = f"Class {class_idx}"
            
            # 绘制边界框
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            
            # 绘制文本背景
            text_width = len(label) * 10
            draw.rectangle([x1, y1, x1+text_width, y1+20], fill="red")
            
            # 绘制文本
            if chinese_font:
                draw.text((x1+2, y1+2), label, font=chinese_font, fill="white")
                # draw.text((x1+2, y1+2), label + str(class_idx), font=chinese_font, fill="white")
            else:
                draw.text((x1+2, y1+2), label, fill="white")
        
        # 保存图像
        img_path = os.path.join(batch_dir, f"image_{img_idx}.png")
        img_pil.save(img_path)
        txt_path=os.path.join(batch_dir, f"image_{img_idx}.json")
        with open(txt_path,'w') as f:
            json.dump(texts_list[img_idx],f,indent=2,ensure_ascii=False)
        print(f"保存可视化图像到: {img_path}")
    
    return batch_dir