#!/usr/bin/env python3
"""
Inference script for Hanuman-o1
"""

import argparse
import torch
import sys
sys.path.append('src')

from training_utils import ReasoningEvaluator
from model import create_hanuman_o1_model

class HanumanO1Inference:
    """Inference class for Hanuman-o1"""

    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load tokenizer
        tokenizer_path = model_path + '_tokenizer'
        self.tokenizer = load_thai_tokenizer(tokenizer_path)

        # Load model
        checkpoint = torch.load(model_path + '.pt', map_location=self.device)
        model_config = checkpoint.get('config', {}).get('model_config', {})

        self.model = create_hanuman_o1_model(**model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def generate_reasoning(self, question, max_length=200):
        """Generate reasoned answer for a question"""
        # Encode question
        input_text = f"คำถาม: {question}\nขั้นตอนการคิด:"
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)

        # Generate reasoning
        with torch.no_grad():
            generated = self.model.generate_reasoning(input_ids, max_length=max_length)

        # Decode result
        result_text = self.tokenizer.decode(generated[0])

        return result_text

    def answer_question(self, question):
        """Simple question answering"""
        reasoning = self.generate_reasoning(question)

        # Extract final answer
        if "คำตอบ:" in reasoning:
            answer = reasoning.split("คำตอบ:")[-1].strip()
        else:
            # Fallback: last sentence
            sentences = reasoning.split('।')
            answer = sentences[-1].strip() if sentences else reasoning

        return {
            'question': question,
            'reasoning': reasoning,
            'answer': answer
        }

def main():
    parser = argparse.ArgumentParser(description='Hanuman-o1 Inference')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--question', type=str,
                       help='Question to answer')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    args = parser.parse_args()

    # Initialize inference
    inference = HanumanO1Inference(args.model_path)

    if args.interactive:
        print("Hanuman-o1 Interactive Mode")
        print("Type 'exit' to quit")
        print("-" * 50)

        while True:
            question = input("Question: ")
            if question.lower() in ['exit', 'quit']:
                break

            result = inference.answer_question(question)
            print(f"Reasoning: {result['reasoning']}")
            print(f"Answer: {result['answer']}")
            print("-" * 50)

    elif args.question:
        result = inference.answer_question(args.question)
        print("Question:", result['question'])
        print("Reasoning:", result['reasoning'])
        print("Answer:", result['answer'])

    else:
        print("Please provide a question with --question or use --interactive mode")

if __name__ == '__main__':
    main()