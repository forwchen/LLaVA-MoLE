# LLaVA-MoLE: Sparse Mixture of LoRA Experts for Mitigating Data Conflicts in Instruction Finetuning MLLMs


## Release
- [2024/03/04] Release model checkpoint and minimal inference code of LLaVA-MoLE.
- [2024/01/30] LLaVA-MoLE paper is on arXiv [[Paper Link](https://arxiv.org/abs/2401.16160)].

## Install
1. Environment
```Shell
conda create -n llava_mole python=3.10 -y
conda activate llava_mole
# install main packages
conda install torch==2.1.2 flash-attn==2.4.2 transformers==4.37.2
```

2. Get Vicuna-v1.5 and patch it for MoE forward
```Shell
cd $CKPT_ROOT # set to directory of checkpoint storage
git-lfs install
git clone https://huggingface.co/lmsys/vicuna-7b-v1.5
git clone https://huggingface.co/openai/clip-vit-large-patch14-336
# go to workspace and clone this repo
git clone https://github.com/forwchen/LLaVA-MoLE
export PROJ_ROOT={PATH_TO_CLONED_DIR} # set PROJ_ROOT to the cloned folder
cp $PROJ_ROOT/moe_patch/modeling_llama.py vicuna-7b-v1.5/modeling_llama.py
```

## Inference
We provide the inference script for testing LLaVA-MoLE checkpoint on Tiny LVLM-eHub.
First, checkout https://github.com/OpenGVLab/Multi-Modality-Arena and add ``llava_mole_infer.py`` to ``tiny_lvlm_evaluation/models``. And download testing data to $LVLM_DATA. Need to also modify ``get_model`` of ``tiny_lvlm_evaluation/models/__init__.py`` to include the following lines:
```Python
    elif model_name == 'LLaVA_MoLE':
        from .llava_mole_infer import TestLLaVAMoLE
        return TestLLaVAMoLE(device)
```
Please rememer to add ``PROJ_ROOT`` to ``PYTHONPATH``. Then run:
```Shell
cd tiny_lvlm_evaluation
python updated_eval_tiny.py --model-name LLaVA_MoLE --device 0 \
    --sampled-root $LVLM_DATA
```


## Citation

If you find this code useful for your research and applications, please cite using this BibTeX:
```bibtex
@article{chen2024llava,
  title={Llava-mole: Sparse mixture of lora experts for mitigating data conflicts in instruction finetuning mllms},
  author={Chen, Shaoxiang and Jie, Zequn and Ma, Lin},
  journal={arXiv preprint arXiv:2401.16160},
  year={2024}
}
```
