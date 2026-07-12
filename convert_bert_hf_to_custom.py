import os
import torch
from transformers import BertForSequenceClassification
from src.model.BERT import BERT

def download_and_convert():
    print("Loading pre-trained 'bert-base-cased' model from Hugging Face...")
    # AG News has 4 classes, so we can initialize the classifier for 4 classes.
    hf_model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=4)
    hf_sd = hf_model.state_dict()

    print("Instantiating custom BERT model...")
    custom_model = BERT() # Custom BERT has 4 output logits by default in layer_id=2 / overall
    custom_sd = custom_model.state_dict()

    new_sd = {}
    mapped_count = 0
    warning_count = 0

    for k, v in custom_sd.items():
        # Map our custom keys to HF keys.
        # Examples:
        # custom: embeddings.word_embeddings.weight -> HF: bert.embeddings.word_embeddings.weight
        # custom: layers.0.attention.self.query.weight -> HF: bert.encoder.layer.0.attention.self.query.weight
        # custom: pooler.dense.weight -> HF: bert.pooler.dense.weight
        # custom: classifier.weight -> HF: classifier.weight
        
        hf_key = None
        
        if k.startswith('classifier.'):
            hf_key = k
        elif k.startswith('embeddings.') or k.startswith('pooler.'):
            hf_key = f"bert.{k}"
        elif k.startswith('layers.'):
            # Replace 'layers.{num}.' with 'bert.encoder.layer.{num}.'
            parts = k.split('.')
            layer_num = parts[1]
            rest = '.'.join(parts[2:])
            hf_key = f"bert.encoder.layer.{layer_num}.{rest}"
            
        if hf_key and hf_key in hf_sd:
            # Check shape matching
            if hf_sd[hf_key].shape == v.shape:
                new_sd[k] = hf_sd[hf_key]
                mapped_count += 1
            else:
                print(f"Warning: Shape mismatch for {k} (custom: {v.shape}, HF: {hf_sd[hf_key].shape}). Keeping custom initialization.")
                new_sd[k] = v
                warning_count += 1
        else:
            print(f"Warning: Key {k} (mapped to {hf_key}) not found in HF state dict. Keeping custom initialization.")
            new_sd[k] = v
            warning_count += 1

    output_path = 'BERT.pt'
    torch.save(new_sd, output_path)
    print(f"\nSuccessfully converted and saved model weights to: {os.path.abspath(output_path)}")
    print(f"Mapped parameters: {mapped_count}")
    print(f"Warnings/Unmapped parameters: {warning_count}")

if __name__ == "__main__":
    download_and_convert()
