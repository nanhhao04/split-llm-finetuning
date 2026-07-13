import torch
import torch.nn as nn
from tqdm import tqdm

# import src.Log
from src.dataset.dataloader import dataloader
from src.model.GPT2 import GPT2

from src.utils.val_GPT2 import simple_generate, compute_bleu, compute_rouge_l


def val_GPT2(state_dict_full, logger, num_val_samples=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Eval device:", device)

    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    test_loader = dataloader(model_name='GPT2', batch_size=4, distribution=[num_val_samples], train=False)

    model = GPT2()
    model.load_state_dict(state_dict_full)
    model = model.to(device)

    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits, _ = model(input_ids, mask)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            if torch.isnan(loss).any():
                print("NaN detected in loss")
                return False, float('inf')

            total_loss += loss.item()

    avg_loss = total_loss / max(len(test_loader), 1)

    logger.log_info(f"Test Loss: {avg_loss:.4f}")

    # Generate samples and compute BLEU / ROUGE-L
    try:
        import os
        import pandas as pd
        from collections import defaultdict
        from transformers import GPT2Tokenizer

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        pad_id = tokenizer.eos_token_id

        devset_path = "./data/E2E/devset.csv"
        if os.path.exists(devset_path):
            test_samples = pd.read_csv(devset_path)
            test_data = []
            for _, row in test_samples.iterrows():
                test_data.append({
                    "mr": row["mr"],
                    "ref": row["ref"]
                })

            mr_to_refs = defaultdict(list)
            for item in test_data:
                mr = item["mr"]
                ref = item["ref"]
                mr_to_refs[mr].append(ref)

            references, predictions = [], []
            count = 0
            # Generate predictions for a maximum of 40 unique MRs to keep it fast
            for mr, refs in tqdm(mr_to_refs.items(), desc="Generating for BLEU/ROUGE-L"):
                if count >= 40:
                    break
                prompt = f"<MR> {mr} </MR> Answer: "
                input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
                attention_mask = torch.ones(input_ids.shape, device=device)

                with torch.no_grad():
                    output_ids = simple_generate(model, input_ids, attention_mask, 50, pad_id, device)
                full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                answer = full_text.split("Answer: ")[-1].strip()

                predictions.append(answer)
                references.append(refs)
                count += 1

            if predictions:
                bleu = compute_bleu(references, predictions)
                rouge_l = compute_rouge_l(references, predictions)

                logger.log_info(f"Test BLEU: {bleu:.4f} | Test ROUGE-L: {rouge_l:.4f}")
                print(f"Test BLEU: {bleu:.4f} | Test ROUGE-L: {rouge_l:.4f}")
    except Exception as e:
        logger.log_info(f"Could not compute BLEU/ROUGE-L: {e}")
        print(f"Could not compute BLEU/ROUGE-L: {e}")

    return True, avg_loss