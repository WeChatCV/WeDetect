from wedetect_ref.models.qwen3vl_referring import Qwen3VLGroundingForConditionalGeneration
from wedetect_ref.models.vision_process import process_vision_info
from transformers import AutoProcessor
from generate_proposal import SimpleYOLOWorldDetector
from vis import plot_bounding_boxes
import argparse
import torch
from PIL import Image
import copy



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--wedetect_ref_checkpoint', type=str, default='')
    parser.add_argument('--wedetect_uni_checkpoint', type=str, default='')
    parser.add_argument('--image', type=str, default='')
    parser.add_argument('--query', type=str, default='')
    parser.add_argument('--score_thre', type=float, default=-1.0)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()


    # load detection model
    model_size = 'base' if 'base' in args.wedetect_uni_checkpoint else 'large'
    det_model = SimpleYOLOWorldDetector(backbone_size=model_size, prompt_dim=768, num_prompts=256, num_proposals=100)
    checkpoint = torch.load(args.wedetect_uni_checkpoint, map_location='cpu')
    # backbone
    keys = list(checkpoint.keys())
    for key in keys:
        if 'backbone' in key:
            new_key = key.replace('backbone.image_model.model.', 'backbone.')
            checkpoint[new_key] = checkpoint.pop(key)
    # head
    keys = list(checkpoint.keys())
    for key in keys:
        if 'bbox_head' in key:
            new_key = key.replace('bbox_head.head_module.', 'bbox_head.')
            new_key = new_key.replace('0.2.', '0.6.')
            new_key = new_key.replace('1.2.', '1.6.')
            new_key = new_key.replace('2.2.', '2.6.')
            new_key = new_key.replace('1.bn', '4')
            new_key = new_key.replace('1.conv', '3')
            new_key = new_key.replace('0.bn', '1')
            new_key = new_key.replace('0.conv', '0')
            checkpoint[new_key] = checkpoint.pop(key)
    det_model = det_model.cuda().eval()
    msg = det_model.load_state_dict(checkpoint, strict=False)


    # load qwen model
    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    model = Qwen3VLGroundingForConditionalGeneration.from_pretrained(args.wedetect_ref_checkpoint, **model_kwargs)
    processor = AutoProcessor.from_pretrained(args.wedetect_ref_checkpoint)
    object_token_index = processor.tokenizer.convert_tokens_to_ids("<object>")
    model.model.object_token_id = object_token_index
    model = model.cuda().eval()


    results = {}
    with torch.no_grad():
        outputs = det_model([args.image])
        results['box'] = outputs[0]['bboxes'].float().cpu().numpy()

    data = {}
    data['id'] = 1
    data['dataset'] = 'refcoco'
    data['image'] = Image.open(args.image).convert('RGB')
    data['proposals'] = results['box'].tolist()
    num_proposals = len(data['proposals'])
    proposal_str = "<object>" * num_proposals
    data['query'] = []
    data['query'].append(
        [{
                "role": "user",
                "content": [{"type": "text", "text": """Please detect the "%s" in the image""" % args.query}]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": proposal_str,}]
            }]
    )

    image = data['image']
    ori_shape = [image.size]
    proposals = [torch.tensor(data['proposals']).cuda().to(model.dtype)]

    for i, prompt in enumerate(data['query']):
        messages = copy.deepcopy(prompt)
        messages[0]['content'] = [{"type": "image", "image": copy.deepcopy(image)}] + messages[0]['content']
        image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)

        texts = [processor.apply_chat_template(messages, tokenize=False)]
        model_inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            do_resize=False,
        )
        model_inputs = model_inputs.to(model.device)
        
        with torch.inference_mode():
            pred = model(
                **model_inputs,
                bboxes=copy.deepcopy(proposals),
                ori_shapes=ori_shape,
                bboxes_id=object_token_index,
                image_inputs=image_inputs,
            )
        
        proposal_positions = model_inputs['input_ids'] == object_token_index
        pred_scores = pred.logits.sigmoid()[proposal_positions].view(-1)
        pred_bboxes = proposals[0].clone().float()
    

    if args.score_thre < 0:
        topk_values, topk_indexes = torch.topk(
            pred_scores.view(-1), 1, dim=0)
        pred_bboxes = pred_bboxes[topk_indexes]
        pred_scores = topk_values
    else:
        mask = pred_scores > args.score_thre
        pred_bboxes = pred_bboxes[mask]
        pred_scores = pred_scores[mask]
    
    if args.visualize:
        pred_image = plot_bounding_boxes(copy.deepcopy(data['image']), pred_bboxes.cpu().tolist())
        pred_image.save("pred.png")  # 你可以自定义保存路径和文件名

        
    

    

