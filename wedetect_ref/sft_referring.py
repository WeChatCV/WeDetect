
import os
import json
import random
import requests
import math
import torch
import torch.nn as nn
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLProcessor,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer import safe_globals, ParallelMode, set_rng_state_for_device
from transformers.utils import is_sagemaker_mp_enabled, logging
import numpy as np
import transformers
from transformers import Trainer
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
)
from models.vision_process import process_vision_info
from models.qwen3vl_referring import Qwen3VLGroundingForConditionalGeneration
from typing import Optional, Tuple
# from datasets import Dataset, DatasetDict
from torch.utils.data import Dataset

from typing import List, Dict, Any
# from torchcodec.decoders import VideoDecoder
import numpy as np
from torchvision.ops.boxes import box_area
from PIL import Image, ImageDraw, ImageFont
import copy
DEFAULT_OBJECT_TOKEN = '<object>'

logger = logging.get_logger(__name__)

def _new_load_rng_state(self, resume_from_checkpoint):
    # Load RNG states from `checkpoint`
    if checkpoint is None:
        return

    if self.args.world_size > 1:
        process_index = self.args.process_index
        rng_file = os.path.join(checkpoint, f"rng_state_{process_index}.pth")
        if not os.path.isfile(rng_file):

            return
    else:
        rng_file = os.path.join(checkpoint, "rng_state.pth")
        if not os.path.isfile(rng_file):
            return

    with safe_globals():
        checkpoint_rng_state = torch.load(rng_file)
    random.setstate(checkpoint_rng_state["python"])
    np.random.set_state(checkpoint_rng_state["numpy"])
    torch.random.set_rng_state(checkpoint_rng_state["cpu"])

    is_distributed = self.args.parallel_mode == ParallelMode.DISTRIBUTED
    if torch.cuda.is_available():
        set_rng_state_for_device("CUDA", torch.cuda, checkpoint_rng_state, is_distributed)


transformers.Trainer._load_rng_state = _new_load_rng_state




from dataclasses import dataclass, field
@dataclass
class SFTModelConfig(ModelConfig):
    freeze_vision_modules: bool = False
    freeze_llm_modules: bool = False
    wedetect: bool = False

@dataclass
class SFTScriptArguments(ScriptArguments):
    data_folder: str = field(
        default="/mnt/data1/shenghao/",
        metadata={"help": "folder to training data"},
    )
    video_sft: Optional[bool] = field(
        default=False,
        metadata={"help": "whether using video_sft"},
    )
    proposal_path: str = field(
        default="proposals.json",
        metadata={"help": "path to proposals json file"},
    )
    multiscale_training: bool = False



# 自定义 Trainer
class CustomTrainer(SFTTrainer):
    def create_optimizer(self):
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        # 如果已经存在优化器，跳过
        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and 'visual' in n)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate * 0.1,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and 'visual' in n)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate * 0.1,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and 'visual' not in n and 'out_proj' not in n)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and 'visual' not in n and 'out_proj' not in n)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and 'out_proj' in n)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate * 10,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and "out_proj" in n)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate * 10,
                },
            ]
            
            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if "bitsandbytes" in str(optimizer_cls) and optimizer_kwargs.get("optim_bits", None) == 8:
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped / 2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2**20}M params")

        if is_sagemaker_mp_enabled():
            import smdistributed.modelparallel.torch as smp
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer
    


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou



class LazySupervisedDataset(Dataset):

    def __init__(self, data_path: str, script_args: SFTScriptArguments):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        with open(data_path, "r") as file:
            self.list_data_dict = json.load(file)
        self.proposals = json.load(open(script_args.proposal_path, "r"))
        self.multiscale_training = script_args.multiscale_training
        # random.shuffle(self.list_data_dict)


    def __len__(self):
        return len(self.list_data_dict)


    def __getitem__(self, i):
        # Format into conversation
        num_base_retries = 5
        try:
            return self._get_item(i)
        except Exception as e:
            print(e)
            print(i)


        for attempt_idx in range(num_base_retries):
            try:
                sample_idx = random.choice(range(len(self)))
                sample = self._get_item(sample_idx)
                return sample
            except Exception as e:
                # no need to sleep
                print(f'[try other #{attempt_idx}] Failed to fetch sample {sample_idx}. Exception:', e)
                pass
        

    def _get_item(self, i):
        source = copy.deepcopy(self.list_data_dict[i])
        # print(source)
        image = Image.open(source["image"]).convert("RGB")
        ori_shape = image.size

        proposals = torch.tensor(self.proposals[source["image"]]).float().reshape(-1, 4)
        gt_boxes = torch.tensor(source['bounding_boxes']).float().reshape(-1, 4)
        grounding_label = torch.zeros(proposals.shape[0])

        if len(gt_boxes) > 0:
            ious = box_iou(gt_boxes, proposals)
            ious = torch.max(ious, dim=1)[0]

            # add gt bboxes
            proposals = torch.cat([proposals, gt_boxes[ious < 0.5]], dim=0)
            proposals = proposals[torch.randperm(proposals.size(0))]
            grounding_label = torch.zeros(proposals.shape[0])

            
            ious = box_iou(gt_boxes, proposals)
            ious = torch.max(ious, dim=0)[0]
            grounding_label[ious > 0.5] = ious[ious > 0.5]


        if 'caption' in source:
            conversations = [
                {'from': 'human', 'value': """<image>\nPlease detect the "%s" described in the caption "%s" in the image""" % (source['class_name'], source['caption'])},
                {'from': 'gpt', 'value': "<object>" * len(proposals)}
            ]
        else:
            conversations = [
                {'from': 'human', 'value': """<image>\nPlease detect the "%s" in the image""" % (source['class_name'])},
                {'from': 'gpt', 'value': "<object>" * len(proposals)}
            ]

        messages = []
        for conv in conversations:
            role = conv["from"]
            content = conv["value"].replace('<image>\n', '')
            if role == "human":
                messages.append({"role": "user", "content": [{"type": "text", "text": content}]})
            elif role == "gpt":
                messages.append({"role": "assistant", "content": [{"type": "text", "text": content}]})
        if self.multiscale_training:
            size = random.uniform(0.5, 1.2)
            min_pixels = int(900 * size) * 32 ** 2
            max_pixels = int(1600 * size) * 32 ** 2
            messages[0]['content'] = [{"type": "image", "image": image, "min_pixels": min_pixels, "max_pixels": max_pixels}] + messages[0]['content']
        else:
            messages[0]['content'] = [{"type": "image", "image": image}] + messages[0]['content']
        image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
        
        texts = [processor.apply_chat_template(messages, tokenize=False)]
        # print(zoomin_intervals, video_sample_fps_list)
        # print(texts)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            do_resize=False,
        )
        # print(inputs["input_ids"].shape)
        if inputs["input_ids"].shape[1] > 5120:
            assert False, 'input too long'
        
        return {
            'image_inputs': image_inputs,
            'video_inputs': video_inputs,
            'messages': messages,
            'ori_shape': ori_shape,
            'bounding_boxes': proposals,
            'bboxes_labels': grounding_label,
        }



def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate batch of examples for training."""
    texts = []
    image_inputs = []
    ori_shapes = []
    bounding_boxes_list = []
    bboxes_labels_list = []

    for i, example in enumerate(examples):
        texts.append(processor.apply_chat_template(example["messages"], tokenize=False))
        image_inputs += example["image_inputs"]
        ori_shapes.append(example['ori_shape'])
        bounding_boxes_list.append(example['bounding_boxes'])
        bboxes_labels_list.append(example['bboxes_labels'])

    inputs = processor(
        text=texts,
        images=image_inputs,
        return_tensors="pt",
        padding=True,
        do_resize=False,
    )

    labels = inputs["input_ids"].clone().float()
    labels[:] = -100

    object_token_index = processor.tokenizer.convert_tokens_to_ids(DEFAULT_OBJECT_TOKEN)
 
    for i, bboxes_label in enumerate(bboxes_labels_list):
        labels[i, inputs["input_ids"][i] == object_token_index] = bboxes_label

    inputs["labels"] = labels
    inputs["ori_shapes"] = ori_shapes
    inputs["bboxes"] = bounding_boxes_list
    inputs["bboxes_labels"] = bboxes_labels_list
    inputs["bboxes_id"] = object_token_index
    inputs['image_inputs'] = image_inputs

    return inputs

if __name__ == "__main__":
    # Parse arguments
    parser = TrlParser((SFTScriptArguments, SFTConfig, SFTModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    
    # Configure training args
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # Setup model
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    # Model initialization
    model_kwargs = dict(
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
    )
    
    model = Qwen3VLGroundingForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    
    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code
    )
    processor.tokenizer.add_tokens([DEFAULT_OBJECT_TOKEN], special_tokens=True)
    object_token_index = processor.tokenizer.convert_tokens_to_ids(DEFAULT_OBJECT_TOKEN)
    model.model.object_token_id = object_token_index

    # Prepare dataset
    dataset = LazySupervisedDataset(script_args.dataset_name, script_args)
    # prepared_dataset = [prepare_dataset(example) for example in dataset['train']]

    if model_config.freeze_vision_modules:
        print("Freezing vision modules...")
        for n, p in model.named_parameters():
            if any(keyword in n for keyword in ['visual', 'wedetect']):
                p.requires_grad = False
    if model_config.freeze_llm_modules:
        print("Freezing LLM modules...")
        for n, p in model.named_parameters():
            if any(keyword in n for keyword in ['language_model']):
                p.requires_grad = False
        model.lm_head.requires_grad = True
    total_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters())
    trainable_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: ~{total_params/1e6:.2f} MB)")
    print(f"Trainable parameters: ~{trainable_params/1e6:.2f} MB)")

    # Initialize trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_config),
        # tokenizer=processor.tokenizer
    )

    # Train model
    if training_args.resume_from_checkpoint is not None:
        checkpoint = get_last_checkpoint(training_args.output_dir)
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # Save final model

    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    if trainer.accelerator.is_main_process:
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # Cleanup
    del model
    del trainer
    torch.cuda.empty_cache()
