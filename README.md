# Hanuman-o1: Thai Language Reasoning AI Model

A state-of-the-art Transformer-based model for Thai language reasoning tasks, designed to achieve exceptional accuracy through innovative architectural components.

## Features

- **Thai-Optimized Architecture**: Custom multi-head attention and positional embeddings for Thai language characteristics
- **Advanced Reasoning**: Integrated chain-of-thought and logical inference capabilities
- **High Accuracy**: Target 100% prediction accuracy on reasoning benchmarks
- **Easy Deployment**: Ready for Hugging Face Hub integration

## Project Structure

```
Hanuman/
├── src/                 # Source code
├── data/                # Datasets and preprocessing
├── models/              # Trained models and checkpoints
├── scripts/             # Training and evaluation scripts
├── notebooks/           # Jupyter notebooks for experimentation
├── config/              # Configuration files
├── architecture_design.md  # Detailed architecture documentation
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Installation

1. Clone the repository
2. Create virtual environment: `python -m venv .venv`
3. Activate environment: `.venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("JonusNattapong/Hanuman-o1")
tokenizer = AutoTokenizer.from_pretrained("JonusNattapong/Hanuman-o1")

# Use for reasoning tasks
```

## Training

Run the training script:

```bash
python scripts/train.py --config config/training_config.json
```

## Evaluation

Evaluate on test sets:

```bash
python scripts/evaluate.py --model_path models/checkpoint-best
```

## Contributing

Contributions welcome! Please read the architecture design document for implementation details.

## License

MIT License
