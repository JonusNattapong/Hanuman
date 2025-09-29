#!/usr/bin/env python3
"""
Main training script for Hanuman-o1
"""

import argparse
import json
import os
import sys
sys.path.append('src')

from tokenizer import create_thai_tokenizer
from model import create_hanuman_o1_model
from data import create_data_loader, prepare_reasoning_data
from training_utils import Trainer, ReasoningEvaluator

def main():
    parser = argparse.ArgumentParser(description='Train Hanuman-o1 model')
    parser.add_argument('--config', type=str, default='config/training_config.json',
                       help='Path to training configuration')
    parser.add_argument('--prepare_data', action='store_true',
                       help='Prepare training data first')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    model_config = config['model_config']
    training_config = config['training_config']

    print("Initializing Hanuman-o1 training...")

    # Create tokenizer
    print("Creating Thai tokenizer...")
    tokenizer = create_thai_tokenizer(vocab_size=model_config['vocab_size'])

    # Prepare data if requested
    if args.prepare_data:
        print("Preparing training data...")
        prepare_reasoning_data(
            'data/raw_reasoning_data.txt',
            training_config['train_file'],
            tokenizer
        )

    # Create model
    print("Creating Hanuman-o1 model...")
    model_config['vocab_size'] = tokenizer.get_vocab_size()
    model = create_hanuman_o1_model(**model_config)

    # Create data loaders
    print("Loading datasets...")
    train_loader = create_data_loader(
        training_config['train_file'],
        tokenizer,
        batch_size=training_config['batch_size']
    )

    val_loader = None
    if os.path.exists(training_config['validation_file']):
        val_loader = create_data_loader(
            training_config['validation_file'],
            tokenizer,
            batch_size=training_config['batch_size'],
            shuffle=False
        )

    # Create trainer
    print("Starting training...")
    trainer = Trainer(model, tokenizer, training_config)
    trainer.train(train_loader, val_loader)

    # Evaluate if test data exists
    if os.path.exists(training_config['test_file']):
        print("Evaluating model...")
        evaluator = ReasoningEvaluator(model, tokenizer)

        with open(training_config['test_file'], 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        accuracy = evaluator.evaluate_reasoning_accuracy(test_data)
        print(f"Final Test Accuracy: {accuracy:.4f}")

    print("Training completed successfully!")

if __name__ == '__main__':
    main()