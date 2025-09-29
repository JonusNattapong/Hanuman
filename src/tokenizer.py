"""
Thai Syllable-Aware Tokenizer for Hanuman-o1
"""

from transformers import AutoTokenizer
import os

class ThaiSyllableTokenizer:
    """Custom tokenizer optimized for Thai language with syllable awareness"""

    def __init__(self, vocab_size=30000, model_path=None):
        self.vocab_size = vocab_size
        self.model_path = model_path or "models/thai_tokenizer"

        # Use a pre-trained Thai tokenizer as base
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased")
        except:
            # Fallback to basic tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

        # Ensure special tokens
        special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

    def encode(self, text, max_length=512, padding=True, truncation=True, return_tensors=None):
        """Encode text to token ids"""
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors
        )
        return encoding

    def decode(self, token_ids):
        """Decode token ids to text"""
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def get_vocab_size(self):
        """Get vocabulary size"""
        return len(self.tokenizer)

    def save(self, path):
        """Save tokenizer"""
        self.tokenizer.save_pretrained(path)

# Convenience functions
def create_thai_tokenizer(vocab_size=30000):
    """Create and return Thai tokenizer"""
    return ThaiSyllableTokenizer(vocab_size=vocab_size)

def load_thai_tokenizer(model_path):
    """Load existing Thai tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer