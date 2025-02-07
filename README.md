# Crammers

Implement, train and fine-tune language models from scratch on consumer grade hardware.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install the dependencies
uv venv
source .venv/bin/activate
uv pip install .
```

## Models

| Model | Dataset Preparation | Training |
|-------|-------------------|-----------|
| BERT  | [Data Preprocessing Pipeline](scripts/data/bert_pretraining_data/README.md) | Coming soon |

## License

Apache 2.0
