import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.ft_llm import (
    build_input,
    build_binary_labels,
    predict_single,
    SegmentClassifier,
    AudioProjection,
    print_trainable_params,
)
from metrics import calculate_metrics
from utils import graph_to_indices


def validate(model, tokenizer, cls_head, audio_proj, val_dataset, config, seg_end_token_id, device):
    """Run validation and return metrics."""
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)
    cls_head.eval()
    if audio_proj is not None:
        audio_proj.eval()

    gts = []
    preds = []

    for row in val_dataset:
        gt = graph_to_indices(row['gt'], config['max_n_gt'], config['gt_threshold'])
        if len(gt) == 0:
            continue
        gts.append(gt)
        preds.append(predict_single(model, tokenizer, cls_head, audio_proj, row, config, seg_end_token_id, device))

    if len(gts) == 0:
        return {'hit': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'ap': 0}
    return calculate_metrics(gts, preds)


def save_checkpoint(save_dir, model, tokenizer, cls_head, audio_proj, config, optimizer, epoch, step):
    """Save full checkpoint."""
    os.makedirs(save_dir, exist_ok=True)

    # save LoRA adapter
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # save classification head
    torch.save(cls_head.state_dict(), os.path.join(save_dir, 'cls_head.pt'))

    # save audio projection if applicable
    if audio_proj is not None:
        torch.save(audio_proj.state_dict(), os.path.join(save_dir, 'audio_proj.pt'))

    # save config
    with open(os.path.join(save_dir, 'ft_llm_config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # save training state for resume
    torch.save({
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
    }, os.path.join(save_dir, 'training_state.pt'))

    print(f"Checkpoint saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train ft_llm model with QLoRA.")
    parser.add_argument('--config', type=str, required=True, help='path to YAML config')
    parser.add_argument('--save_dir', type=str, required=True, help='checkpoint output directory')
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint dir to resume from')
    parser.add_argument('--dev', action='store_true', help='use dev dataset')
    args = parser.parse_args()

    # load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # mutual exclusion check
    if config.get('use_dva', False) and config.get('use_hubert', False):
        raise ValueError("Cannot use both DVA and HuBERT features simultaneously.")

    print(f"Config: {config}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load dataset
    from unsloth import FastLanguageModel

    dataset = load_dataset("yhpark/rhapsody-dev" if args.dev else "yhpark/rhapsody")
    dataset.set_format("numpy")
    train_dataset = dataset['train']
    val_dataset = dataset['val']

    # load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config['model_name'],
        max_seq_length=config['max_seq_length'],
        load_in_4bit=True,
    )

    # apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # handle end-of-segment token
    seg_end_token = config['seg_end_token']
    if seg_end_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({'additional_special_tokens': [seg_end_token]})
        model.resize_token_embeddings(len(tokenizer))
    seg_end_token_id = tokenizer.convert_tokens_to_ids(seg_end_token)

    # determine classifier input dim
    hidden_dim = 2048
    use_dva = config.get('use_dva', False)
    use_hubert = config.get('use_hubert', False)
    hubert_layer = config.get('hubert_layer', 9)

    if use_dva:
        cls_input_dim = hidden_dim + 1024
    elif use_hubert:
        cls_input_dim = hidden_dim + 768
    else:
        cls_input_dim = hidden_dim

    # create classification head and audio projection
    cls_head = SegmentClassifier(cls_input_dim).to(device)
    audio_proj = None
    if use_dva:
        audio_proj = AudioProjection(1024).to(device)
    elif use_hubert:
        audio_proj = AudioProjection(768).to(device)

    # collect trainable parameters
    trainable_params = list(model.parameters()) + list(cls_head.parameters())
    if audio_proj is not None:
        trainable_params += list(audio_proj.parameters())

    # filter to only trainable params (LoRA)
    trainable_params = [p for p in trainable_params if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )

    # resume from checkpoint
    start_epoch = 0
    start_step = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        model.load_adapter(args.resume, adapter_name="default")
        cls_head.load_state_dict(torch.load(os.path.join(args.resume, 'cls_head.pt'), map_location=device))
        if audio_proj is not None:
            audio_proj_path = os.path.join(args.resume, 'audio_proj.pt')
            if os.path.exists(audio_proj_path):
                audio_proj.load_state_dict(torch.load(audio_proj_path, map_location=device))
        training_state_path = os.path.join(args.resume, 'training_state.pt')
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location=device)
            optimizer.load_state_dict(training_state['optimizer_state_dict'])
            start_epoch = training_state['epoch']
            start_step = training_state['step']
            print(f"Resumed at epoch {start_epoch}, step {start_step}")

    print_trainable_params(model, cls_head, audio_proj)

    loss_fn = nn.BCEWithLogitsLoss()
    grad_accum_steps = config['grad_accum_steps']
    best_avg_metric = -1.0

    for epoch in range(start_epoch, config['epochs']):
        model.train()
        cls_head.train()
        if audio_proj is not None:
            audio_proj.train()

        optimizer.zero_grad()
        epoch_loss = 0.0
        n_steps = 0

        for step, row in enumerate(tqdm(train_dataset, desc=f"Epoch {epoch+1}/{config['epochs']}")):
            # skip already-processed steps when resuming
            if epoch == start_epoch and step < start_step:
                continue

            text = build_input(row['segment_summaries'], seg_end_token)
            inputs = tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=config['max_seq_length'],
            ).to(device)

            outputs = model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]

            seg_positions = (inputs['input_ids'][0] == seg_end_token_id).nonzero(as_tuple=False)
            segment_reprs = last_hidden[0, seg_positions[:, 0], :]

            # audio fusion
            if use_dva and audio_proj is not None:
                audio_feat = torch.tensor(np.array(row['dva']), dtype=torch.float32).to(device)
                audio_feat = audio_proj(audio_feat)
                segment_reprs = torch.cat([segment_reprs, audio_feat], dim=-1)
            elif use_hubert and audio_proj is not None:
                hubert_feat = torch.tensor(
                    np.array(row['hubert'][:, hubert_layer, :]),
                    dtype=torch.float32,
                ).to(device)
                hubert_feat = audio_proj(hubert_feat)
                segment_reprs = torch.cat([segment_reprs, hubert_feat], dim=-1)

            logits = cls_head(segment_reprs.float()).squeeze(-1)
            labels = build_binary_labels(row['gt'], config['max_n_gt'], config['gt_threshold']).to(device)
            loss = loss_fn(logits, labels) / grad_accum_steps
            loss.backward()

            epoch_loss += loss.item() * grad_accum_steps
            n_steps += 1

            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # handle remaining gradients
        if n_steps % grad_accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss / max(n_steps, 1)
        print(f"Epoch {epoch+1} - avg loss: {avg_loss:.4f}")

        # validate
        print("Validating...")
        val_metrics = validate(
            model, tokenizer, cls_head, audio_proj,
            val_dataset, config, seg_end_token_id, device,
        )
        print(f"Validation metrics: {val_metrics}")

        # compute average metric for checkpoint selection (average of all evaluation metrics)
        avg_metric = (val_metrics['hit'] + val_metrics['precision'] + val_metrics['recall'] + val_metrics['f1'] + val_metrics['ap']) / 5

        # save if best
        if avg_metric > best_avg_metric:
            best_avg_metric = avg_metric
            save_checkpoint(
                args.save_dir, model, tokenizer, cls_head, audio_proj,
                config, optimizer, epoch + 1, 0,
            )
            print(f"New best avg metric: {best_avg_metric:.4f}")

    print(f"Training complete. Best avg metric: {best_avg_metric:.4f}")


if __name__ == '__main__':
    main()
