"""
Data loading and preprocessing for Hanuman-o1
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForLanguageModeling
import pandas as pd
import json
from typing import List, Dict, Any
import os

class ThaiReasoningDataset(Dataset):
    """Dataset for Thai reasoning tasks"""

    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(data_path)

    def load_data(self, data_path):
        """Load reasoning data from file"""
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            data = df.to_dict('records')
        else:
            # Assume text file with one example per line
            with open(data_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            data = [{'text': line.strip()} for line in lines if line.strip()]

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Handle different data formats
        if 'question' in item and 'answer' in item:
            # Reasoning format: question + chain-of-thought + answer
            text = f"คำถาม: {item['question']}\nขั้นตอนการคิด: {item.get('reasoning', '')}\nคำตอบ: {item['answer']}"
        elif 'text' in item:
            text = item['text']
        else:
            text = str(item)

        # Tokenize
        encoding = self.tokenizer.encode(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()  # For language modeling
        }

class ReasoningDataCollator:
    """Data collator for reasoning tasks"""

    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def __call__(self, batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])

        # Apply masked language modeling for reasoning tasks
        if self.mlm_probability > 0:
            input_ids, labels = self.mask_tokens(input_ids, labels)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    def mask_tokens(self, inputs, labels):
        """Prepare masked tokens inputs/labels for reasoning tasks"""
        probability_matrix = torch.full(inputs.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in inputs.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(inputs.shape, 0.8)).bool() & masked_indices
        mask_token_id = self.tokenizer.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenizer.mask_token)
        inputs[indices_replaced] = mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(inputs.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer.tokenizer), inputs.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

def create_data_loader(data_path, tokenizer, batch_size=8, max_length=512, shuffle=True):
    """Create data loader for training/evaluation"""
    dataset = ThaiReasoningDataset(data_path, tokenizer, max_length)
    data_collator = ReasoningDataCollator(tokenizer)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=data_collator
    )

def prepare_reasoning_data(raw_data_path, output_path, tokenizer):
    """Prepare reasoning dataset from raw data"""
    # This function would preprocess raw Thai reasoning data
    # For now, create sample data

    sample_data = [
        {
            "question": "ถ้าทุกคนในห้องมีอายุมากกว่า 20 ปี และสมชายอายุ 25 ปี สมชายอยู่ในห้องหรือไม่",
            "reasoning": "ทุกคนในห้องมีอายุมากกว่า 20 ปี และสมชายอายุ 25 ปี ซึ่งมากกว่า 20 ปี ดังนั้นสมชายอยู่ในห้อง",
            "answer": "ใช่ สมชายอยู่ในห้อง"
        },
        {
            "question": "น้ำหนักของสินค้าคือ 2 กิโลกรัม ราคาต่อกิโลกรัมคือ 50 บาท ราคารวมเท่าไร",
            "reasoning": "น้ำหนัก = 2 กิโลกรัม, ราคาต่อกิโลกรัม = 50 บาท, ราคารวม = 2 × 50 = 100 บาท",
            "answer": "100 บาท"
        }
    ]

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)

    print(f"Prepared reasoning data saved to {output_path}")

# Utility functions
def load_json_data(file_path):
    """Load JSON data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_data(data, file_path):
    """Save data as JSON"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)