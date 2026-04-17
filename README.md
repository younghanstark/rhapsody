# Rhapsody: A Dataset for Highlight Detection in Podcasts

This is the official repository for the paper, "Rhapsody: A Dataset for Highlight Detection in Podcasts", presented in COLM 2025.

[![arXiv](https://img.shields.io/badge/📝%20arXiv-orange.svg)](https://arxiv.org/abs/2505.19429)
[![Dataset (Text-Only)](https://img.shields.io/badge/🤗%20Dataset-yellow.svg)](https://huggingface.co/datasets/yhpark/rhapsody)

Feel free to ask questions by opening a new issue or sending email to younghanpark@yonsei.ac.kr.

## Table of Contents

1. [Installation](#1-installation)
2. [Usage](#2-usage)
3. [Citation](#3-citation)

## 1. Installation

```bash
conda create -n rhapsody python=3.13
conda activate rhapsody
pip install datasets rank-bm25 openai google-genai
```

Before running the code, please request access to our [dataset](https://huggingface.co/datasets/yhpark/rhapsody) on Hugging Face. Once access is granted, log in to your Hugging Face account using `hf auth login`.

To use zero-shot prompting with language models, set the appropriate API key environment variable:
- **OpenAI models** (e.g., `gpt-4o-2024-08-06`): Set `OPENAI_API_KEY`
- **Gemini models** (e.g., `gemini-2.0-flash-001`): Set `GEMINI_API_KEY`

## 2. Usage

```
python evaluate.py [evaluation options] <method> [method options]
```

### Evaluation Options

| Option | Default | Description |
|--------|---------|-------------|
| `--max_n_gt` | 20 | Maximum number of GT highlight segments to extract |
| `--gt_threshold` | 0.5 | Minimum replay score for a segment to be a highlight |
| `--n_trials` | 1 | Number of evaluation runs (results are averaged) |

### Available Methods

| Method | Description |
|--------|-------------|
| `random` | Random sampling baseline. Samples random segments based on the GT distribution. |
| `frequency` | Frequency-based baseline. Predicts the most frequently highlighted segments from the train split. |
| `bm25` | BM25-based detection. Ranks segments by BM25 similarity to the podcast title. |
| `zeroshot` | Zero-shot prompting with LLMs (GPT-4o, Gemini). |
| `ft_llm` | Fine-tuned LLM with QLoRA (Llama-3.2-1B-Instruct) and segment-level classification. |

See each method's `add_cli_args` class method for a list of its specific command-line options.

Note: The `--use_audio` option for the `zeroshot` method tries to provide the original podcast audio from YouTube, but may not work in all examples, as some podcast episodes may be unavailable, hidden, or removed from YouTube.

### Fine-tuned LLMs with Segment-Level Classification Heads (`ft_llm`)

#### Additional Setup

```bash
pip install unsloth torch peft transformers pyyaml
```

Requires a CUDA-capable GPU for 4-bit QLoRA training and inference.

#### Training

```bash
# text-only
python ft_llm/train.py --config ft_llm/configs/text_only.yaml --save_dir checkpoints/text_only

# text + DVA
python ft_llm/train.py --config ft_llm/configs/text_dva.yaml --save_dir checkpoints/text_dva

# text + HuBERT
python ft_llm/train.py --config ft_llm/configs/text_hubert.yaml --save_dir checkpoints/text_hubert
```

#### Resume Training

```bash
python ft_llm/train.py --config ft_llm/configs/text_only.yaml --save_dir checkpoints/text_only --resume checkpoints/text_only
```

#### Evaluation

```bash
# text-only
python evaluate.py ft_llm --checkpoint_path checkpoints/text_only

# with audio features
python evaluate.py ft_llm --checkpoint_path checkpoints/text_dva --use_dva
python evaluate.py ft_llm --checkpoint_path checkpoints/text_hubert --use_hubert
```

#### Trainable Parameters

| Setting | LoRA | Classifier | Audio Proj | Total |
|---------|------|-----------|------------|-------|
| text-only | ~5.6M | 2,049 | — | ~5.6M |
| text+DVA | ~5.6M | 3,073 | 1,049,600 | ~6.7M |
| text+HuBERT | ~5.6M | 2,817 | 590,592 | ~6.2M |

## 3. Citation

If you find our work helpful, please cite our work as follows:
```bibtex
@inproceedings{park2025rhapsody,
    title={Rhapsody: A Dataset for Highlight Detection in Podcasts}, 
    author={Younghan Park and Anuj Diwan and David Harwath and Eunsol Choi},
    year={2025},
    booktitle={Conference on Language Modeling (COLM) 2025},
    url={https://arxiv.org/abs/2503.04713},
}
```
