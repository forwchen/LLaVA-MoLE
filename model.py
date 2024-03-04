import os
import logging
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from copy import deepcopy
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai_vit import CLIPVisionTower, process_images
from moe import LoRA_MOE_LM
from peft import get_peft_model, TaskType, LoraConfig


class MLLM_InputAdaptor_Vicuna(nn.Module):
    def __init__(self, args, in_channel, out_channel, text_embed, tokenizer):
        super().__init__()
        self.args = args
        self.adaptor_in_channel = in_channel
        self.adaptor_out_channel = out_channel

        self.visual_start_token = nn.Parameter(torch.randn(1,1, self.adaptor_out_channel)) #1*1*4096
        self.visual_end_token = nn.Parameter(torch.randn(1,1, self.adaptor_out_channel)) #1*1*4096
        self.zero = torch.zeros([self.adaptor_out_channel], dtype=torch.bfloat16)

        self.text_embed = text_embed
        self.tokenizer = tokenizer
        self.context_len = tokenizer.model_max_length


        mlp_depth = 2
        modules = [nn.Linear(in_channel, out_channel)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(out_channel, out_channel))
        self.projector = nn.Sequential(*modules)

        self._init_parameters()
        self.user_token_ids = tokenizer("USER:", add_special_tokens=False, return_tensors="pt").input_ids[0]
        self.asst_token_ids = tokenizer("ASSISTANT:", add_special_tokens=False, return_tensors="pt").input_ids[0]

    def _init_parameters(self):
        token_std = (self.adaptor_out_channel) ** -0.5
        nn.init.normal_(self.visual_start_token.data, std=token_std)
        nn.init.normal_(self.visual_end_token.data, std=token_std)

    def do_not_train(self, tensors):
        if isinstance(tensors, list):
            for x in tensors:
                x = x.detach()
        else:
            tensors = tensors.detach()
        return tensors

    def forward(self, img_tokens, local_tokens, instructions, answers):
        batch_size = len(instructions)
        device = instructions[0].device
        
        instruction_embeds = []
        instruction_ids = []
        instruction_l = []
        for text in instructions:
            # split instruction into chunks by image token
            img_idx = np.where(text.cpu().numpy() == 99999)[0].tolist()
            sections = np.array(img_idx+[text.size(0)]) - np.array([0]+img_idx)
            chunks = torch.split(text, sections.tolist())

            # get chunk embedings
            instruction_chunks = [chunks[0]]
            inst_l = chunks[0].size(0)
            for c in chunks[1:]:
                instruction_chunks.append(c[1:])
                inst_l += c[1:].size(0)
            tmp = [self.text_embed(text) for text in instruction_chunks]
            tmp = self.do_not_train(tmp)

            instruction_embeds.append(tmp)
            instruction_ids.append(instruction_chunks)
            instruction_l.append(inst_l)

        proj_img_tokens = []
        for i, (_, img_token) in enumerate(img_tokens):
            img_token = self.projector(img_token)
            if len(local_tokens) > 0: # is highres inputs
                img_token = torch.cat([self.visual_start_token, 
                                        img_token, 
                                        local_tokens[i].reshape(1, -1, self.adaptor_out_channel),
                                        self.visual_end_token], 1)
            else:
                img_token = torch.cat([self.visual_start_token, 
                                        img_token, 
                                        self.visual_end_token], 1)
            proj_img_tokens.append(img_token)
            instruction_l[i] += img_token.size(1)

        if answers is None: # inference mode, assume bs=1
            input_embeds = []
            for i in range(proj_img_tokens[0].size(0)):
                input_embeds.append(instruction_embeds[0][i])
                input_embeds.append(proj_img_tokens[0][i])
            input_embeds.append(instruction_embeds[0][-1])
            input_embeds = torch.cat(input_embeds, 0).unsqueeze(0)    
            return input_embeds

        answer_embeds = [self.text_embed(text) for text in answers]
        answer_embeds = self.do_not_train(answer_embeds)

        answer_l = [_.shape[0] for _ in answer_embeds]
        
        empty_tgt_id = -100

        max_length = -1
        for i in range(batch_size):
            max_length = max(max_length, instruction_l[i]+answer_l[i]+1)
        
        if max_length > self.context_len:
            logging.info(f"inputs length {max_length}={instruction_l}+{answer_l} > context length {self.context_len}")

        input_embeds = self.text_embed(torch.ones([batch_size, max_length], 
                                                      dtype=torch.long).to(device).fill_(self.tokenizer.pad_token_id))
        input_embeds = self.do_not_train(input_embeds)
        input_targets = torch.ones([batch_size, max_length],
                                    dtype=torch.long).to(device).fill_(empty_tgt_id)
        
        attention_masks = torch.zeros([batch_size, max_length], dtype=torch.long).to(device)

        loss_masks = torch.zeros([batch_size, max_length], dtype=torch.long).to(device)

        for i in range(batch_size):
            has_valid_img = img_tokens[i][0]
            
            assert len(instruction_embeds[i]) == proj_img_tokens[i].size(0)+1

            input_targets[i, :instruction_l[i]] = empty_tgt_id
            attention_masks[i, :instruction_l[i]+answer_l[i]] = 1
            ids = instruction_ids[i]

            cur_l = 0
            for j in range(len(instruction_embeds[i])):
                inst_emb = instruction_embeds[i][j]
                if j < len(instruction_embeds[i])-1:
                    img_emb = proj_img_tokens[i][j]

                input_embeds[i, cur_l:cur_l+inst_emb.size(0), :] = inst_emb

                if j < len(instruction_embeds[i])-1:
                    input_embeds[i, cur_l+inst_emb.size(0):cur_l+inst_emb.size(0)+img_emb.size(0), :] = img_emb
                    if not has_valid_img:
                        attention_masks[i, cur_l+inst_emb.size(0):cur_l+inst_emb.size(0)+img_emb.size(0)] = 0
            
                
                user_id_starts = self.find_id_starts(ids[j].cpu().numpy(), self.user_token_ids.numpy())
                asst_id_starts = self.find_id_starts(ids[j].cpu().numpy(), self.asst_token_ids.numpy())
                if len(asst_id_starts) > 0:
                    if len(user_id_starts) > 0 and user_id_starts[0] < asst_id_starts[0]:
                        asst_id_starts = asst_id_starts[1:]
                    for k in range(len(asst_id_starts)-1):
                        st = cur_l + asst_id_starts[k]+len(self.asst_token_ids)
                        ed = cur_l + user_id_starts[k]
                        input_targets[i, st:ed] = ids[j][st-cur_l:ed-cur_l]
                        loss_masks[i, st:ed] = 1

                cur_l += inst_emb.size(0)
                if j < len(instruction_embeds[i])-1:
                    cur_l += img_emb.size(0)

            input_embeds[i, instruction_l[i]:instruction_l[i]+answer_l[i], :] = answer_embeds[i]
            input_targets[i, instruction_l[i]:instruction_l[i]+answer_l[i]] = answers[i]
            loss_masks[i, instruction_l[i]:instruction_l[i]+answer_l[i]] = 1
        
        return input_embeds, input_targets, attention_masks, loss_masks

    def find_id_starts(self, seq, ids):
        def rolling_window(a, size):
            shape = a.shape[:-1] + (a.shape[-1] - size + 1, size)
            strides = a.strides + (a. strides[-1],)
            return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        
        bool_indices = np.all(rolling_window(seq, len(ids)) == ids, axis=1)
        return np.mgrid[0:len(bool_indices)][bool_indices]

class MLLM_LLaVA_MoLE(nn.Module):
    def __init__(self, args, device, is_train, llm_model, tokenizer, vit):
        super().__init__()
        self.args = args
        self.is_train = is_train
        self.device = device
        self.llm_model = llm_model
        self.tokenizer = tokenizer

        for n, p in self.llm_model.named_parameters():
            p.requires_grad = False

        if self.args.use_lora:
            print("Using lora...")
            peft_cfg = LoraConfig(
                            task_type=TaskType.CAUSAL_LM, 
                            inference_mode=not self.is_train, 
                            r=args.lora_rank, 
                            lora_alpha=args.lora_alpha, 
                            lora_dropout=0.05,
                            target_modules=self.args.lora_modules.split(','))
            self.llm_model = get_peft_model(self.llm_model, peft_cfg)

        if self.args.llm_moe:
            assert self.args.use_lora
            assert  'gate_proj' not in self.args.lora_modules and \
                    'up_proj' not in self.args.lora_modules and \
                    'down_proj' not in self.args.lora_modules

            num_layers = len(self.llm_model.base_model.model.model.layers)
            for i in range(num_layers):
                original_mlp = self.llm_model.base_model.model.model.layers[i].mlp
                self.llm_model.base_model.model.model.layers[i].mlp = \
                    LoRA_MOE_LM(args=args,
                        lora_rank=self.args.lora_rank,
                        lora_alpha=self.args.lora_alpha,
                        num_experts=args.llm_moe_num_experts,
                        original_module=original_mlp).bfloat16()

        self.vit = vit
        vit_trainable_param = []
        for param in args.vit_trainable.split(','):
            
            layer_prefix = 'vision_tower.vision_model.encoder.layers'

            if not param.startswith(layer_prefix):
                vit_trainable_param.append(param)
            else:
                layers = param.split('.')[-1]
                if layers.startswith('[') and layers.endswith(']'):
                    st, ed = layers[1:-1].split('-')
                    for i in range(int(st), int(ed)+1):
                        vit_trainable_param.append(layer_prefix+'.'+str(i))
                else:
                    vit_trainable_param.append(param)

        vit_not_trainable_param = ['position_embedding', 'patch_embedding', 'class_embedding', 'pos_embed']
        for vit_param_name, p in self.vit.named_parameters():
            trainable = False
            for nt in vit_trainable_param:
                if vit_param_name.startswith(nt):
                    trainable = True
            for nt in vit_not_trainable_param:
                if nt in vit_param_name:
                    trainable = False
            p.requires_grad = trainable

        llm_hidden_chn = llm_model.lm_head.in_features
        vit_out_chn = vit.vision_tower.vision_model.post_layernorm.weight.size(0)

        self.in_adaptor = MLLM_InputAdaptor_Vicuna(
            args,
            in_channel=vit_out_chn,
            out_channel=llm_hidden_chn,
            text_embed=llm_model.model.embed_tokens,
            tokenizer=tokenizer,
        )

    def forward(self, image_tensors, instructions, answers, is_text_only):
        img_tokens = []
        local_tokens = []

        for img_t, is_text in zip(image_tensors, is_text_only):
            global_token = self.vit(img_t)
            img_tokens.append((not is_text, global_token))

        batch_size = len(instructions)
        inputs_embeds, input_targets, attention_masks, loss_masks = self.in_adaptor(img_tokens, local_tokens, instructions, answers)
        ret_dict = self.llm_model(inputs_embeds=inputs_embeds, labels=input_targets, attention_mask=attention_masks)
        text_loss = ret_dict['loss']
        mask = loss_masks[:, 1:].to(text_loss.dtype)
        text_loss = (text_loss.view(batch_size, -1) * mask).sum() / mask.sum()
        
        if self.args.llm_moe and self.args.llm_moe_num_experts > 1:
            llm_mlp_routing_probs = torch.stack([r[0] for r in ret_dict.routings], dim=0) # [layer, batch, seq_len, num_experts]
            llm_mlp_routing_idxes = torch.stack([r[1] for r in ret_dict.routings], dim=0).detach()

            llm_mlp_expert_balancing_loss = 0.
            for i in range(batch_size):
                probs_i = llm_mlp_routing_probs[:,i, attention_masks[i].bool()].reshape(-1, self.args.llm_moe_num_experts)
                idxes_i = llm_mlp_routing_idxes[:,i, attention_masks[i].bool()].reshape(-1, self.args.llm_moe_num_experts)

                llm_mlp_expert_balancing_loss += (probs_i.mean(0) * idxes_i.mean(0)).sum()

            text_loss += llm_mlp_expert_balancing_loss/batch_size * self.args.moe_balance_w

        return text_loss


    def generate(self, 
                 image_tensors, 
                 instructions, 
                 do_sample=True, 
                 temperature=0.5,
                 num_beams=1, 
                 repetition_penalty=10.0, 
                 length_penalty=1):
        instructions = instructions.unsqueeze(0)
        img_tokens = self.vit(image_tensors.unsqueeze(0))
        inputs_embeds = self.in_adaptor([(True, img_tokens)], [], instructions, answers=None)
        output_ids = self.llm_model.generate(
                inputs_embeds=inputs_embeds.bfloat16(),
                max_new_tokens=256,
                do_sample=do_sample,
                temperature=temperature,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty)[0].cpu()
        if output_ids[0] == self.tokenizer.bos_token_id:
            output_ids = output_ids[1:]
        if output_ids[-1] == self.tokenizer.eos_token_id:
            output_ids = output_ids[:-1]
        return self.tokenizer.decode(output_ids, add_special_tokens=False)


def create_model_llava_mole(
    args,
    is_train: bool = True,
    device: torch.device = torch.device('cpu'),
    ):
    CKPT_ROOT = os.environ.get("CKPT_ROOT", None)
    vicuna_path = f"{CKPT_ROOT}/vicuna-7b-v1.5/"
    vit_path = f"{CKPT_ROOT}/clip-vit-large-patch14-336/"
    
    tokenizer = AutoTokenizer.from_pretrained(vicuna_path, trust_remote_code=True)
    tokenizer.model_max_length = 4096
    model = AutoModelForCausalLM.from_pretrained(
                vicuna_path, device_map='cpu', 
                torch_dtype=torch.bfloat16, trust_remote_code=True)
    vit = CLIPVisionTower(args, vit_path)
    img_transform = partial(process_images, image_processor=vit.image_processor)

    model = MLLM_LLaVA_MoLE(args, device, is_train, llm_model=model, tokenizer=tokenizer, vit=vit)
    
    return model, img_transform, tokenizer



