import torch
import torch.nn as nn
import torchvision
import re
import math
import copy
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLPreTrainedModel, Qwen3VLModel, is_torchdynamo_compiling, Cache, Qwen3VLModelOutputWithPast, Qwen3VLCausalLMOutputWithPast
from typing import Optional, List, Union, Tuple
import torch.nn.functional as F
import torch.distributed as dist

def gen_sineembed_for_position(pos_tensor, embedding_dim):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    dim = embedding_dim // pos_tensor.size(-1)
    scale = 2 * math.pi
    dim_t = torch.arange(dim, dtype=pos_tensor.dtype, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / dim)
    x_embed = pos_tensor[:, 0] * scale
    y_embed = pos_tensor[:, 1] * scale
    pos_x = x_embed[:, None] / dim_t
    pos_y = y_embed[:, None] / dim_t
    pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
    pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=1)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, 2] * scale
        pos_w = w_embed[:, None] / dim_t
        pos_w = torch.stack((pos_w[:, 0::2].sin(), pos_w[:, 1::2].cos()), dim=2).flatten(1)

        h_embed = pos_tensor[:, 3] * scale
        pos_h = h_embed[:, None] / dim_t
        pos_h = torch.stack((pos_h[:, 0::2].sin(), pos_h[:, 1::2].cos()), dim=2).flatten(1)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=1)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)




def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean()





class Qwen3VLModelGrounding(Qwen3VLModel):
    def __init__(self, config):
        super().__init__(config)
        # 这里可以添加新的模块，例如 bbox 编码器
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', 'mlp2x_gelu')
        mlp_depth = int(mlp_gelu_match.group(1))


        modules = [nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size))
        self.image_pos_projector = nn.Sequential(*modules)
        self.image_pos_projector[-1].weight.data.zero_()
        self.image_pos_projector[-1].bias.data.zero_()

        print(config.text_config.hidden_size)
        if config.text_config.hidden_size > 4000: 
            modules = [nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size))
            self.object_vision_projector = nn.Sequential(*modules)
        else:
            modules = [nn.Linear(config.text_config.hidden_size * 7 * 7, config.text_config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size))
            self.object_vision_projector = nn.Sequential(*modules)

        modules = [nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size))
        self.object_pos_projector = nn.Sequential(*modules)
        self.object_pos_projector[-1].weight.data.zero_()
        self.object_pos_projector[-1].bias.data.zero_()

        self.second_scale_conv = nn.ConvTranspose2d(config.text_config.hidden_size, config.text_config.hidden_size // 2, kernel_size=2, stride=2)
        self.first_scale_conv1 = nn.ConvTranspose2d(config.text_config.hidden_size, config.text_config.hidden_size // 2, kernel_size=2, stride=2)
        self.first_scale_norm = nn.LayerNorm(config.text_config.hidden_size // 2)
        self.first_scale_act = nn.GELU()
        self.first_scale_conv2 = nn.ConvTranspose2d(config.text_config.hidden_size // 2, config.text_config.hidden_size // 4, kernel_size=2, stride=2)
        self.merge = nn.Linear(config.text_config.hidden_size // 4 + config.text_config.hidden_size // 2 + config.text_config.hidden_size, config.text_config.hidden_size)


    def generate_coordinate(self, featmap, device='cuda'):
        featmap_sizes = featmap.shape[-2:]
        x_range = torch.linspace(0, int(featmap_sizes[1])-1, int(featmap_sizes[1]), device=device) / int(featmap_sizes[1])
        y_range = torch.linspace(0, int(featmap_sizes[0])-1, int(featmap_sizes[0]), device=device) / int(featmap_sizes[0])
        y, x = torch.meshgrid(y_range, x_range)
        y = y.unsqueeze(-1)
        x = x.unsqueeze(-1)
        coord_feat = torch.cat([x, y], -1)

        return coord_feat
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        bboxes = None,  # 新增参数：边界框
        ori_shapes = None,  # 新增参数：原始图像尺寸
        **kwargs,
    ) -> Union[tuple, Qwen3VLModelOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_features = []
            object_features = []
            object_masks = []
            split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
            scale3_image_feats = [feat.clone() for feat in image_embeds]
            scale2_image_feats = torch.split(deepstack_image_embeds[-1], split_sizes)
            scale1_image_feats = torch.split(deepstack_image_embeds[-2], split_sizes)
            # extract RoI features based on bboxes
            for i, (scale1_image_feat, scale2_image_feat, scale3_image_feat, ori_shape, feat_shape, bbox) in enumerate(zip(scale1_image_feats, scale2_image_feats, scale3_image_feats, ori_shapes, image_grid_thw, bboxes)):
                feat_shape = feat_shape.cpu().tolist()
                T, H, W = feat_shape
                H = H // self.visual.spatial_merge_size
                W = W // self.visual.spatial_merge_size
                # the first scale
                scale1_image_feat = scale1_image_feat.reshape(T, H, W, self.config.text_config.hidden_size).permute(0, 3, 1, 2).contiguous()
                scale1_image_feat = self.first_scale_conv1(scale1_image_feat).permute(0, 2, 3, 1)
                scale1_image_feat = self.first_scale_act(self.first_scale_norm(scale1_image_feat)).permute(0, 3, 1, 2)
                scale1_image_feat = self.first_scale_conv2(scale1_image_feat)

                # the second scale
                scale2_image_feat = scale2_image_feat.reshape(T, H, W, self.config.text_config.hidden_size).permute(0, 3, 1, 2).contiguous()
                scale2_image_feat = self.second_scale_conv(scale2_image_feat)

                # the third scale
                scale3_image_feat = scale3_image_feat.reshape(T, H, W, self.config.text_config.hidden_size).permute(0, 3, 1, 2).contiguous()

                if len(bbox) == 0:
                    gt_bbox = torch.tensor([[0, 0, W * 32, H * 32]], device=scale3_image_feat.device, dtype=scale3_image_feat.dtype)
                    object_masks.append(torch.tensor([0], device=scale3_image_feat.device, dtype=torch.bool))
                else:
                    gt_bbox = torch.tensor(bbox, device=scale3_image_feat.device, dtype=scale3_image_feat.dtype) / (torch.tensor([ori_shape[0], ori_shape[1], ori_shape[0], ori_shape[1]], device=scale3_image_feat.device, dtype=scale3_image_feat.dtype) / torch.tensor([W * 32, H * 32, W * 32, H * 32], device=scale3_image_feat.device, dtype=scale3_image_feat.dtype))
                    object_masks.append(torch.tensor([1]*len(bbox), device=scale3_image_feat.device, dtype=torch.bool))
                
                roi_feats1 = torchvision.ops.roi_align(scale1_image_feat.float(), [gt_bbox.float()], 7, 1/8).to(scale3_image_feat.dtype)
                roi_feats2 = torchvision.ops.roi_align(scale2_image_feat.float(), [gt_bbox.float()], 7, 1/16).to(scale3_image_feat.dtype)
                roi_feats3 = torchvision.ops.roi_align(scale3_image_feat.float(), [gt_bbox.float()], 7, 1/32).to(scale3_image_feat.dtype)

                # image_feats
                image_coor = (self.generate_coordinate(scale3_image_feat) + 0.5).to(scale3_image_feat.dtype)
                image_coor = self.image_pos_projector(gen_sineembed_for_position(image_coor.flatten(0, 1), self.config.text_config.hidden_size))
                image_features.append(image_embeds[i] + image_coor)

                # object_feats
                roi_feats = torch.cat([roi_feats1, roi_feats2, roi_feats3], dim=1).permute(0, 2, 3, 1)
                roi_feats = self.merge(roi_feats)
                if self.config.text_config.hidden_size > 4000: 
                    roi_feats = roi_feats.flatten(1, 2)
                    roi_feats = torch.mean(roi_feats, dim=1)
                    roi_feats = self.object_vision_projector(roi_feats)
                else:
                    roi_feats = self.object_vision_projector(roi_feats.flatten(1))
                box_coor = box_xyxy_to_cxcywh(gt_bbox) / torch.tensor([W * 32, H * 32, W * 32, H * 32], device=gt_bbox.device, dtype=gt_bbox.dtype)
                box_coor = self.object_pos_projector(gen_sineembed_for_position(box_coor, self.config.text_config.hidden_size))
                object_features.append(roi_feats + box_coor)

            image_embeds = torch.cat(image_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            object_masks = torch.cat(object_masks, dim=0)
            object_id_mask = (input_ids == self.object_token_id).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(object_id_mask, torch.cat(object_features, dim=0)[object_masks])

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            # aggregate visual_pos_masks and deepstack_visual_embeds
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                # Only apply conversion for floating point tensors (inverted masks)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )


# 自定义扩展类
class Qwen3VLGroundingForConditionalGeneration(Qwen3VLForConditionalGeneration):
    def __init__(self, config: Qwen3VLConfig):
        super().__init__(config)
        self.model = Qwen3VLModelGrounding(config)
        # self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.out_proj = nn.Linear(config.text_config.hidden_size, 1)
        # nn.init.xavier_uniform_(self.out_proj.weight)
        # nn.init.zeros_(self.out_proj.bias)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.out_proj.bias.data, bias_value)

        self.post_init()


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        bboxes = None,  # 新增参数：边界框
        ori_shapes = None,  # 新增参数：原始图像尺寸
        bboxes_labels = None,  # 新增参数：边界框标签
        bboxes_id = None,  # 新增参数：边界框对应的 token id
        **kwargs,
    ) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.

        Example:
            TODO: Add example
        """
        proposal_positions = input_ids == bboxes_id
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            bboxes=bboxes,  # 传递边界框
            ori_shapes=ori_shapes,  # 传递原始图像尺寸
            **kwargs,
        )

        
        batch_size = input_ids.shape[0]
        hidden_states = outputs[0]
        logits_batch = self.out_proj(hidden_states)
        proposal_logits = []
        proposal_labels = []
        for b in range(batch_size):
            proposal_logits.append(logits_batch[b][proposal_positions[b]])
        proposal_logits = torch.cat(proposal_logits, dim=0).float().view(-1)

        loss = None
        if bboxes_labels is not None:
            bboxes_labels = torch.cat(bboxes_labels).to(proposal_logits.device).float().view(-1)

            # grounding_labels = bboxes_labels.clone()
            # grounding_labels[grounding_labels > 0] = 1
            # positive_samples = torch.sum(grounding_labels)
            # negative_samples = grounding_labels.numel() - positive_samples
            # pos_weight = torch.sqrt(negative_samples / max(1, positive_samples)).to(hidden_states.device)

            # loss_fct = nn.BCEWithLogitsLoss(pos_weight=torch.min(torch.tensor(5.0, device=hidden_states.device), pos_weight))

            # loss = loss_fct(proposal_logits, grounding_labels)

            positive_samples = torch.sum(bboxes_labels > 0).float()
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(positive_samples)
            positive_samples = torch.clamp(positive_samples / get_world_size(), min=1).item()

            loss = sigmoid_focal_loss(proposal_logits, bboxes_labels, positive_samples)

        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits_batch,
            past_key_values=outputs.past_key_values,
            rope_deltas=outputs.rope_deltas,
        )
    