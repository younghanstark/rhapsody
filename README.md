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

See each method's `add_cli_args` class method for a list of its specific command-line options.

**Important Note**: The code for `fine-tuned LLMs with segment-level classification heads` will be available soon :)

Note: The `--use_audio` option for the `zeroshot` method tries to provide the original podcast audio from YouTube, but may not work in all examples, as some podcast episodes may be unavailable, hidden, or removed from YouTube.

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
