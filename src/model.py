"""
Hanuman-o1 Model Architecture
Thai Language Reasoning Transformer with Chain-of-Thought capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class ThaiOptimizedMultiHeadAttention(nn.Module):
    """Multi-head attention with Thai-specific optimizations"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        # Thai-specific relative positional embeddings
        self.relative_pos_emb = nn.Embedding(512, d_model // n_heads)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # Linear transformations and reshape
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Add relative positional bias for Thai
        seq_len = q.size(-2)
        pos_ids = torch.arange(seq_len, device=q.device).unsqueeze(0)
        rel_pos_bias = self.relative_pos_emb(pos_ids - pos_ids.transpose(-1, -2))
        scores = scores + rel_pos_bias.unsqueeze(1)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, v)

        # Concatenate heads and put through final linear layer
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(context)

        return output

class FeedForward(nn.Module):
    """Feed-forward network with Thai-specific activations"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        # Use GELU for Thai language processing
        self.activation = nn.GELU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class TransformerBlock(nn.Module):
    """Transformer encoder block with Thai optimizations"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = ThaiOptimizedMultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-head attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x

class ReasoningLayer(nn.Module):
    """Dedicated reasoning layer for logical inference"""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attention = ThaiOptimizedMultiHeadAttention(d_model, n_heads)
        self.norm = nn.LayerNorm(d_model)

        # Logical operation embeddings
        self.logic_ops = nn.Embedding(4, d_model)  # AND, OR, NOT, IMPLIES

    def forward(self, x, logic_mask=None):
        # Apply logical reasoning attention
        reasoned = self.attention(x, x, x, logic_mask)
        return self.norm(x + reasoned)

class ChainOfThoughtGenerator(nn.Module):
    """Generates intermediate reasoning steps"""

    def __init__(self, d_model, vocab_size, max_steps=5):
        super().__init__()
        self.max_steps = max_steps
        self.step_generator = nn.Linear(d_model, vocab_size)
        self.reasoning_proj = nn.Linear(d_model, d_model)

    def forward(self, hidden_states, reasoning_tokens):
        """Generate chain of thought steps"""
        cot_steps = []

        for step in range(self.max_steps):
            # Project to reasoning space
            reasoning_input = self.reasoning_proj(hidden_states)

            # Generate next reasoning token
            logits = self.step_generator(reasoning_input[:, -1, :])
            next_token = torch.argmax(logits, dim=-1)

            # Update hidden states with reasoning token
            # (Simplified - in practice, use decoder or additional processing)
            cot_steps.append(next_token)

        return torch.stack(cot_steps, dim=1)

class HanumanO1Model(nn.Module):
    """Main Hanuman-o1 model for Thai reasoning"""

    def __init__(self, vocab_size, d_model=768, n_layers=12, n_heads=12, d_ff=3072,
                 max_seq_len=512, dropout=0.1, n_reasoning_layers=3):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token and positional embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Reasoning layers
        self.reasoning_layers = nn.ModuleList([
            ReasoningLayer(d_model, n_heads)
            for _ in range(n_reasoning_layers)
        ])

        # Chain-of-thought generator
        self.cot_generator = ChainOfThoughtGenerator(d_model, vocab_size)

        # Output head
        self.lm_head = nn.Linear(d_model, vocab_size)

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None, labels=None):
        seq_len = input_ids.size(1)

        # Create position ids
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        # Embeddings
        token_emb = self.token_emb(input_ids)
        pos_emb = self.pos_emb(pos_ids)
        x = self.dropout(token_emb + pos_emb)

        # Create attention mask
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            mask = None

        # Encoder layers
        for layer in self.encoder_layers:
            x = layer(x, mask)

        # Reasoning layers
        for reasoning_layer in self.reasoning_layers:
            x = reasoning_layer(x, mask)

        # Apply layer norm
        x = self.norm(x)

        # Generate logits
        logits = self.lm_head(x)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return {'logits': logits, 'loss': loss, 'hidden_states': x}

    def generate_reasoning(self, input_ids, max_length=100):
        """Generate response with chain-of-thought reasoning"""
        with torch.no_grad():
            # Encode input
            outputs = self.forward(input_ids)
            hidden_states = outputs['hidden_states']

            # Generate reasoning steps
            reasoning_tokens = self.cot_generator(hidden_states, input_ids)

            # Combine input and reasoning for final output
            combined_input = torch.cat([input_ids, reasoning_tokens], dim=1)

            # Generate final answer
            for _ in range(max_length - combined_input.size(1)):
                outputs = self.forward(combined_input)
                next_token_logits = outputs['logits'][:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                combined_input = torch.cat([combined_input, next_token], dim=1)

                # Stop if EOS token (assuming 2 is EOS)
                if next_token.item() == 2:
                    break

        return combined_input

def create_hanuman_o1_model(vocab_size, **kwargs):
    """Factory function to create Hanuman-o1 model"""
    return HanumanO1Model(vocab_size, **kwargs)