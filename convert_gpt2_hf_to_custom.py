import os
import torch
from transformers import GPT2LMHeadModel
from src.model.GPT2 import GPT2


def download_and_convert():
    print("Loading pre-trained 'gpt2' model from Hugging Face...")
    hf_model = GPT2LMHeadModel.from_pretrained('gpt2')
    hf_sd = hf_model.state_dict()

    print("Instantiating custom GPT-2 model (layer_id=0, full model)...")
    custom_model = GPT2(layer_id=0, n_block=12)
    custom_sd = custom_model.state_dict()

    new_sd = {}
    mapped_count = 0
    warning_count = 0

    for k, v in custom_sd.items():
        # Mapping custom keys → HuggingFace keys:
        #
        # Embeddings
        #   wte.weight              → transformer.wte.weight
        #   wpe.weight              → transformer.wpe.weight
        #
        # Transformer blocks
        #   h.{i}.ln_1.*            → transformer.h.{i}.ln_1.*
        #   h.{i}.ln_2.*            → transformer.h.{i}.ln_2.*
        #   h.{i}.attn.c_attn.*    → transformer.h.{i}.attn.c_attn.*
        #   h.{i}.attn.c_proj.*    → transformer.h.{i}.attn.c_proj.*
        #   h.{i}.mlp.c_fc.*       → transformer.h.{i}.mlp.c_fc.*
        #   h.{i}.mlp.c_proj.*     → transformer.h.{i}.mlp.c_proj.*
        #
        # Final LayerNorm & LM head
        #   ln_f.*                  → transformer.ln_f.*
        #   lm_head.weight          → lm_head.weight

        hf_key = None

        if k == 'lm_head.weight':
            hf_key = 'lm_head.weight'
        elif k.startswith('wte.') or k.startswith('wpe.') or k.startswith('ln_f.'):
            hf_key = f'transformer.{k}'
        elif k.startswith('h.'):
            hf_key = f'transformer.{k}'

        if hf_key and hf_key in hf_sd:
            if hf_sd[hf_key].shape == v.shape:
                new_sd[k] = hf_sd[hf_key]
                mapped_count += 1
            else:
                print(
                    f"Warning: Shape mismatch for {k!r} "
                    f"(custom: {v.shape}, HF: {hf_sd[hf_key].shape}). "
                    f"Keeping custom initialization."
                )
                new_sd[k] = v
                warning_count += 1
        else:
            print(
                f"Warning: Key {k!r} (mapped to {hf_key!r}) not found in HF state dict. "
                f"Keeping custom initialization."
            )
            new_sd[k] = v
            warning_count += 1

    output_path = 'GPT2.pt'
    torch.save(new_sd, output_path)
    print(f"\nSuccessfully converted and saved model weights to: {os.path.abspath(output_path)}")
    print(f"Mapped parameters:            {mapped_count}")
    print(f"Warnings/Unmapped parameters: {warning_count}")


if __name__ == "__main__":
    download_and_convert()
