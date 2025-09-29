#!/usr/bin/env python3
"""
Evaluation script for Hanuman-o1
"""

import argparse
import json
import torch
import sys
sys.path.append('src')

from tokenizer import load_thai_tokenizer
from model import create_hanuman_o1_model
from training_utils import ReasoningEvaluator

def main():
    parser = argparse.ArgumentParser(description='Evaluate Hanuman-o1 model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--test_file', type=str, default='data/test.json',
                       help='Path to test data')
    parser.add_argument('--output_file', type=str, default='results/evaluation_results.json',
                       help='Path to save evaluation results')
    args = parser.parse_args()

    print("Loading model and tokenizer...")

    # Load tokenizer
    tokenizer = load_thai_tokenizer(args.model_path + '_tokenizer')

    # Load model configuration
    with open(args.model_path + '.pt', 'rb') as f:
        checkpoint = torch.load(f, map_location='cpu')
        model_config = checkpoint.get('config', {}).get('model_config', {})

    # Create model
    model = create_hanuman_o1_model(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create evaluator
    evaluator = ReasoningEvaluator(model, tokenizer)

    # Load test data
    print("Loading test data...")
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # Evaluate
    print("Evaluating reasoning accuracy...")
    accuracy = evaluator.evaluate_reasoning_accuracy(test_data)

    # Save results
    results = {
        'accuracy': accuracy,
        'num_samples': len(test_data),
        'model_path': args.model_path,
        'test_file': args.test_file
    }

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Evaluation completed. Results saved to {args.output_file}")
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    main()