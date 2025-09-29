"""
Training utilities for Hanuman-o1
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from tqdm import tqdm
import os
import json
from typing import Dict, Any

class Trainer:
    """Trainer class for Hanuman-o1 model"""

    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # Setup optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 5e-5),
            weight_decay=config.get('weight_decay', 0.01)
        )

        # Setup scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('num_epochs', 10) * config.get('steps_per_epoch', 1000)
        )

        # Setup wandb if enabled
        if config.get('use_wandb', False):
            wandb.init(project="hanuman-o1", config=config)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Best model tracking
        self.best_loss = float('inf')
        self.best_model_path = None

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            outputs = self.model(**batch)
            loss = outputs['loss']

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / num_batches

    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs['loss']

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train(self, train_loader, val_loader=None):
        """Full training loop"""
        num_epochs = self.config.get('num_epochs', 10)

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_loss = self.train_epoch(train_loader)
            print(f"Train Loss: {train_loss:.4f}")

            # Validate
            val_loss = None
            if val_loader:
                val_loss = self.validate(val_loader)
                print(f"Val Loss: {val_loss:.4f}")

            # Log to wandb
            if self.config.get('use_wandb', False):
                log_data = {'epoch': epoch + 1, 'train_loss': train_loss}
                if val_loss:
                    log_data['val_loss'] = val_loss
                wandb.log(log_data)

            # Save best model
            current_loss = val_loss if val_loss else train_loss
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.save_model(f"models/checkpoint-epoch-{epoch+1}")

        print("Training completed!")

    def save_model(self, path):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_loss': self.best_loss
        }, path + '.pt')

        # Save tokenizer
        self.tokenizer.save(path + '_tokenizer')

        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path + '.pt')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_loss = checkpoint.get('best_loss', float('inf'))

        print(f"Model loaded from {path}")

class ReasoningEvaluator:
    """Evaluator for reasoning tasks"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def evaluate_reasoning_accuracy(self, test_data):
        """Evaluate reasoning accuracy"""
        correct = 0
        total = 0

        with torch.no_grad():
            for item in tqdm(test_data, desc="Evaluating"):
                question = item['question']
                expected_answer = item['answer']

                # Generate reasoning
                input_ids = self.tokenizer.encode(question, return_tensors='pt').to(self.device)
                generated = self.model.generate_reasoning(input_ids, max_length=200)

                # Decode generated text
                generated_text = self.tokenizer.decode(generated[0])

                # Extract answer (simplified - look for "คำตอบ:" pattern)
                if "คำตอบ:" in generated_text:
                    predicted_answer = generated_text.split("คำตอบ:")[-1].strip()
                else:
                    predicted_answer = generated_text.split()[-1]  # Last token as fallback

                # Check accuracy (simplified string matching)
                if expected_answer.lower() in predicted_answer.lower():
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0
        print(f"Reasoning Accuracy: {accuracy:.4f} ({correct}/{total})")
        return accuracy

def create_trainer(model, tokenizer, config_path):
    """Create trainer from config"""
    with open(config_path, 'r') as f:
        config = json.load(f)

    return Trainer(model, tokenizer, config)

def load_config(config_path):
    """Load training configuration"""
    with open(config_path, 'r') as f:
        return json.load(f)