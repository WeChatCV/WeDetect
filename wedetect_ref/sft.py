
import os
import json
import random
import requests
import math
import torch
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
import numpy as np
import transformers
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
from models.qwen3vl_grounding import Qwen3VLGroundingForConditionalGeneration
from typing import Optional, Tuple
# from datasets import Dataset, DatasetDict
from torch.utils.data import Dataset

from typing import List, Dict, Any
# from torchcodec.decoders import VideoDecoder
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import copy
DEFAULT_OBJECT_TOKEN = '<object>'

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



class LazySupervisedDataset(Dataset):

    def __init__(self, data_path: str, script_args: SFTScriptArguments):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        with open(data_path, "r") as file:
            self.list_data_dict = json.load(file)
        # random.shuffle(self.list_data_dict)


    def __len__(self):
        return len(self.list_data_dict)


    def __getitem__(self, i):
        # Format into conversation
        num_base_retries = 3
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

        messages = []
        for conv in source["conversations"]:
            role = conv["from"]
            content = conv["value"].replace('<image>\n', '')
            if role == "human":
                messages.append({"role": "user", "content": [{"type": "text", "text": content}]})
            elif role == "gpt":
                messages.append({"role": "assistant", "content": [{"type": "text", "text": content}]})
        messages[0]['content'] = [{"type": "image", "image": image}] + messages[0]['content']
        image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
        
        bounding_boxes = source.get("bounding_boxes", [])
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
            'bounding_boxes': bounding_boxes,
        }


def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate batch of examples for training."""
    texts = []
    image_inputs = []
    ori_shapes = []
    bounding_boxes_list = []

    for i, example in enumerate(examples):
        texts.append(processor.apply_chat_template(example["messages"], tokenize=False))
        image_inputs += example["image_inputs"]
        ori_shapes.append(example['ori_shape'])
        bounding_boxes_list.append(example['bounding_boxes'])

    inputs = processor(
        text=texts,
        images=image_inputs,
        return_tensors="pt",
        padding=True,
        do_resize=False,
    )

    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    object_token_index = processor.tokenizer.convert_tokens_to_ids(DEFAULT_OBJECT_TOKEN)
    visual_tokens = [151652, 151653, 151656, 151655, object_token_index] # ['<|vision_start|><|vision_end|><|video_pad|><|image_pad|>']

    for visual_token_id in visual_tokens:
        labels[labels == visual_token_id] = -100

    inputs["labels"] = labels
    inputs["ori_shapes"] = ori_shapes
    inputs["bboxes"] = bounding_boxes_list
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
    trainer = SFTTrainer(
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
