import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from model import create_model_llava_mole
from vicuna_conversation import build_chat_input

class LLaVA_MoLE_Args:
    vit_trainable = ''
    dense_moe = True
    use_lora = True
    llm_moe = True
    lora_rank = 32
    lora_alpha = 64
    lora_modules = 'q_proj,k_proj,v_proj,o_proj'
    llm_moe_num_experts = 3


class TestLLaVAMoLE:
    def __init__(self, device=None):
        args = LLaVA_MoLE_Args()
        self.model, self.transform, self.tokenizer = create_model_llava_mole(args, False, device)

        model_ckpt_path = os.path.join(os.environ.get("CKPT_ROOT", None), 'llava_mole_r16_a32_e3.pt')
        checkpoint = torch.load(model_ckpt_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        msg = self.model.load_state_dict(state_dict, strict=False)
        self.model.to(device)
        self.model.eval()
        self.model.bfloat16()
        self.device = device

    def ask_and_answer(self, 
                    pil_image, question, 
                    do_sample=False, 
                    temperature=0.5, 
                    num_beams=1, 
                    repetition_penalty=10.0, 
                    length_penalty=1):
        messages = []
        messages.append({"role": "user", "content": question})
        chat_input = build_chat_input(self.model.tokenizer, messages)[0]
        chat_input = chat_input.to(self.device)

        img_tensor = self.transform(pil_image).to(self.device).bfloat16()

        output_text = self.model.generate(img_tensor, chat_input, 
                                    do_sample=do_sample, 
                                    temperature=temperature,
                                    num_beams=num_beams,
                                    repetition_penalty=repetition_penalty,
                                    length_penalty=length_penalty)

        return output_text

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=256):
        pil_image = Image.open(image).convert('RGB')
        instruction = 'Analyze the image and provide brief and accurate answer to the question in Engligh. '

        question = instruction + question
        ans = self.ask_and_answer(pil_image, question)
        return ans
    
    @torch.no_grad()
    def pure_generate(self, image, question, max_new_tokens=256):
        return self.generate(image, question, max_new_tokens)

    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=256):
        batch_outputs = [self.generate(image, question) for image, question in zip(image_list, question_list)]
        return batch_outputs

