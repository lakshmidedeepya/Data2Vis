# %% [code] {"execution":{"iopub.status.busy":"2025-07-17T09:39:22.322950Z","iopub.execute_input":"2025-07-17T09:39:22.323655Z","iopub.status.idle":"2025-07-17T09:39:22.340593Z","shell.execute_reply.started":"2025-07-17T09:39:22.323631Z","shell.execute_reply":"2025-07-17T09:39:22.340016Z"}}
import json
from typing import List, Dict, Tuple
from dateutil.parser import parse
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
import json
from collections import Counter
class DataPreprocessor:
    """
    Handles field name transformations and character-level vocabularies.
    Builds source and target vocabs dynamically from data.
    """

    def __init__(self):
        self.source_char_to_idx = {}
        self.source_idx_to_char = {}
        self.target_char_to_idx = {}
        self.target_idx_to_char = {}

    def is_date(self, string: str) -> bool:
        try:
            if string.isdigit():
                return False
            parse(string)
            return True
        except Exception:
            return False

    def transform_field_names(self, data_str: str) -> Tuple[str, Dict[str, str]]:
        try:
            data = json.loads(data_str)

            if not data:
                print(f"Data is empty or not a list: {data}")
                return data_str, {}

            field_mapping = {}
            str_counter, num_counter, dt_counter = 0, 0, 0
            
            for field, value in data.items():
                
                value_str = str(value).strip()
                
                if self.is_date(value_str):
                    
                    generic_name = f"dt{dt_counter}"
                    dt_counter += 1
                else:
                    try:
                        
                        float_val = float(value_str)
                        # optionally distinguish between int and float
                        generic_name = f"num{num_counter}"
                        num_counter += 1
                    except ValueError:
                        
                        generic_name = f"str{str_counter}"
                        str_counter += 1
                field_mapping[field] = generic_name
                
            # Replace keys in original string (only keys, not values!)
            transformed = data_str
            for original, generic in field_mapping.items():
                transformed = transformed.replace(f'"{original}"', f'"{generic}"')
            print(f"Transformed field names in transform_field_names: {field_mapping}")
            return transformed, field_mapping
        except Exception:
            return data_str, {}

    def reverse_transform_field_names(self, spec_str: str, field_mapping: Dict[str, str]) -> str:
        reversed_mapping = {v: k for k, v in field_mapping.items()}
        for generic, original in reversed_mapping.items():
            spec_str = spec_str.replace(f'"{generic}"', f'"{original}"')
        return spec_str

    def build_vocabularies(self, source_texts: List[str], target_texts: List[str], min_freq: int = 5):
        """
        Build character-level vocabularies with frequency filtering.
        Only characters appearing more than `min_freq` times are included.
        """
    
        # Build source vocab
        source_char_counts = Counter()
        for text in source_texts:
            if isinstance(text, str):  # Safety check
                source_char_counts.update(text)
        
        source_chars = [char for char, count in source_char_counts.items() if count >= min_freq]
        source_chars = sorted(set(source_chars))  # Remove duplicates and sort
        source_chars = ['<PAD>', '<UNK>'] + source_chars
        
        self.source_char_to_idx = {char: idx for idx, char in enumerate(source_chars)}
        self.source_idx_to_char = {idx: char for char, idx in self.source_char_to_idx.items()}
        
        # Build target vocab
        target_char_counts = Counter()
        for text in target_texts:
            if isinstance(text, str):  # Safety check
                target_char_counts.update(text)
        
        target_chars = [char for char, count in target_char_counts.items() if count >= min_freq]
        target_chars = sorted(set(target_chars))  # Remove duplicates and sort
        target_chars = ['<PAD>', '<UNK>', '<START>', '<END>'] + target_chars
        
        self.target_char_to_idx = {char: idx for idx, char in enumerate(target_chars)}
        self.target_idx_to_char = {idx: char for char, idx in self.target_char_to_idx.items()}
        
        print(f"Source vocab size: {len(self.source_char_to_idx)}")
        print(f"Target vocab size: {len(self.target_char_to_idx)}")

    def text_to_sequence(self, text: str, vocab: Dict[str, int], max_length: int = None) -> List[int]:
        if not isinstance(text, str):
            print(f"Warning: Expected string, got {type(text)}")
            text = str(text)
        
        sequence = []
        for char in text:
            if char in vocab:
                sequence.append(vocab[char])
            else:
                sequence.append(vocab.get('<UNK>', 1))  # Safe fallback
        
        if max_length and len(sequence) > max_length:
            sequence = sequence[:max_length]
        
        return sequence

    def sequence_to_text(self, sequence: List[int], idx_to_char: Dict[int, str]) -> str:
        chars = []
        for idx in sequence:
            char = idx_to_char.get(idx, '')
            if char in ['<PAD>', '<UNK>', '<START>', '<END>']:
                if char == '<END>':
                    break
                continue
            chars.append(char)
        return ''.join(chars)


# %% [code] {"execution":{"iopub.status.busy":"2025-07-17T09:39:22.341850Z","iopub.execute_input":"2025-07-17T09:39:22.342104Z","iopub.status.idle":"2025-07-17T09:39:22.367955Z","shell.execute_reply.started":"2025-07-17T09:39:22.342081Z","shell.execute_reply":"2025-07-17T09:39:22.367438Z"}}
class Data2VisDataset(Dataset):
    def __init__(self,source_sequences, target_sequences, preprocessor, max_src_len=500, max_tgt_len=500):
        self.sources = source_sequences
        self.targets = target_sequences
        self.preprocessor = preprocessor
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        # Fixed: sources and targets are already sequences
        src_seq = self.sources[idx]
        tgt_seq = self.targets[idx]
        if len(src_seq)>self.max_src_len:
            src_seq=src_seq[:self.max_src_len]
        if len(tgt_seq)>self.max_tgt_len:
            tgt_seq=tgt_seq[:self.max_tgt_len]

        return {
            'source': torch.tensor(src_seq, dtype=torch.long),
            'target': torch.tensor(tgt_seq, dtype=torch.long),
        }

# %% [code] {"execution":{"iopub.status.busy":"2025-07-17T09:39:22.368637Z","iopub.execute_input":"2025-07-17T09:39:22.368862Z","iopub.status.idle":"2025-07-17T09:39:22.393193Z","shell.execute_reply.started":"2025-07-17T09:39:22.368846Z","shell.execute_reply":"2025-07-17T09:39:22.392688Z"}}
import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    Collate function for Data2Vis dataset that pads sequences.
    Expects batch items as dicts: {'source': tensor, 'target': tensor, ...}
    """
    # Extract sources and targets
    source_seqs = [item['source'] for item in batch]
    target_seqs = [item['target'] for item in batch]

    # Pad source and target sequences
    padded_sources = pad_sequence(source_seqs, batch_first=True, padding_value=0)  # 0 = <PAD> index
    padded_targets = pad_sequence(target_seqs, batch_first=True, padding_value=0)

    return {
        'source': padded_sources,           # [B, max_src_len]
        'target': padded_targets,           # [B, max_tgt_len]
        'source_lengths': torch.tensor([len(seq) for seq in source_seqs]),
        'target_lengths': torch.tensor([len(seq) for seq in target_seqs])
    }


# %% [code] {"execution":{"iopub.status.busy":"2025-07-17T09:39:22.393947Z","iopub.execute_input":"2025-07-17T09:39:22.394136Z","iopub.status.idle":"2025-07-17T09:39:26.667687Z","shell.execute_reply.started":"2025-07-17T09:39:22.394122Z","shell.execute_reply":"2025-07-17T09:39:26.666954Z"}}
# Step 1: Load source and target text lines from Data/Q3
split = "train"  # or "dev" or "test"
src_file_path = f'{split}.sources'
tgt_file_path = f'{split}.targets'

with open(src_file_path, 'r', encoding='utf-8') as f:
    train_sources_raw = [line.strip() for line in f]

with open(tgt_file_path, 'r', encoding='utf-8') as f:
    train_targets_raw = [line.strip() for line in f]

preprocessor = DataPreprocessor()
preprocessor.build_vocabularies(train_sources_raw, train_targets_raw, min_freq=100)

train_sources_sequences = []
for src in train_sources_raw:
    seq = preprocessor.text_to_sequence(src, preprocessor.source_char_to_idx, max_length=1000)
    train_sources_sequences.append(seq)

train_targets_sequences = []
for tgt in train_targets_raw:
    # Add START and END tokens
    seq = [preprocessor.target_char_to_idx['<START>']]
    seq.extend(preprocessor.text_to_sequence(tgt, preprocessor.target_char_to_idx, max_length=1000))
    seq.append(preprocessor.target_char_to_idx['<END>'])
    train_targets_sequences.append(seq)
# Step 2: Create Dataset and DataLoader
train_dataset = Data2VisDataset(train_sources_sequences, train_targets_sequences, preprocessor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True,collate_fn=collate_fn)

# save vocab to file
import json
with open("source_vocab.json", "w") as f:
    json.dump(preprocessor.source_char_to_idx, f)
with open("target_vocab.json", "w") as f:
    json.dump(preprocessor.target_char_to_idx, f)

# %% [code] {"execution":{"iopub.status.busy":"2025-07-17T09:39:26.669717Z","iopub.execute_input":"2025-07-17T09:39:26.669922Z","iopub.status.idle":"2025-07-17T09:39:26.674274Z","shell.execute_reply.started":"2025-07-17T09:39:26.669906Z","shell.execute_reply":"2025-07-17T09:39:26.673666Z"}}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# %% [code] {"execution":{"iopub.status.busy":"2025-07-17T09:39:26.675006Z","iopub.execute_input":"2025-07-17T09:39:26.675185Z","iopub.status.idle":"2025-07-17T09:39:26.702422Z","shell.execute_reply.started":"2025-07-17T09:39:26.675169Z","shell.execute_reply":"2025-07-17T09:39:26.701743Z"}}
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionLayer(nn.Module):
    """
    Dot-product attention mechanism layer for the sequence-to-sequence model.
    """

    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.scale = math.sqrt(hidden_size)

    def forward(self, query, values, mask=None):
        # query shape: (batch_size, hidden_size)
        # values shape: (batch_size, max_length, hidden_size)

        batch_size, max_length, _ = values.size()

        # Reshape query to (batch_size, 1, hidden_size) for bmm
        query = query.unsqueeze(1)  # (batch_size, 1, hidden_size)

        # Compute attention scores with dot product: (B, 1, T)
        attention_scores = torch.bmm(query, values.transpose(1, 2)) / self.scale  # (B, 1, T)

        # Squeeze to remove middle dimension -> (B, T)
        attention_scores = attention_scores.squeeze(1)

        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e10)

        # Attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # (B, T)

        # Compute context vector as weighted sum of values
        context_vector = torch.bmm(attention_weights.unsqueeze(1), values)  # (B, 1, H)
        context_vector = context_vector.squeeze(1)  # (B, H)

        return context_vector, attention_weights


# %% [code] {"execution":{"iopub.status.busy":"2025-07-17T09:39:26.703162Z","iopub.execute_input":"2025-07-17T09:39:26.703425Z","iopub.status.idle":"2025-07-17T09:39:26.725711Z","shell.execute_reply.started":"2025-07-17T09:39:26.703400Z","shell.execute_reply":"2025-07-17T09:39:26.725118Z"}}
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
class EncoderLSTM(nn.Module):
    """
    Bidirectional LSTM encoder.
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout_rate):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size, num_layers,
            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout_rate)
        # Add projection layer for bidirectional outputs
        self.output_projection = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, x, lengths):
        # x shape: (batch_size, sequence_length)
        embedded = self.dropout(self.embedding(x))
        
        if lengths.device != torch.device('cpu'):
            lengths = lengths.cpu()
        lengths = lengths.to(torch.int64)

        # Pack padded sequence
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        
        # Forward through LSTM
        packed_output, (hidden, cell) = self.lstm(packed)
        
        # Unpack sequence
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Combine forward and backward hidden states
        # hidden shape: (num_layers * 2, batch_size, hidden_size)
        batch_size = hidden.size(1)
        
        # Take the last layer's hidden states
        forward_hidden = hidden[-2, :, :]  # Forward direction
        backward_hidden = hidden[-1, :, :]  # Backward direction
        
        # Concatenate forward and backward hidden states
        final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # Project bidirectional outputs back to hidden_size
        output = self.output_projection(output)
        return output, final_hidden

# %% [code] {"execution":{"iopub.status.busy":"2025-07-17T09:39:26.726428Z","iopub.execute_input":"2025-07-17T09:39:26.727250Z","iopub.status.idle":"2025-07-17T09:39:26.749331Z","shell.execute_reply.started":"2025-07-17T09:39:26.727226Z","shell.execute_reply":"2025-07-17T09:39:26.748780Z"}}
class Decoder(nn.Module):
    """
    LSTM decoder with attention mechanism.
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout_rate):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size, num_layers,
            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0
        )
        self.attention = AttentionLayer(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Output projection layer
        self.out = nn.Linear(hidden_size * 2, vocab_size)  # *2 for concatenated context
        
    def forward(self, x, hidden, cell, encoder_outputs, encoder_mask=None):
        # x shape: (batch_size, 1)
        embedded = self.dropout(self.embedding(x))
        
        # Forward through LSTM
        lstm_output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        
        # Apply attention
        context, attention_weights = self.attention(
            lstm_output.squeeze(1), encoder_outputs, encoder_mask
        )
        
        # Concatenate LSTM output with context
        concat_output = torch.cat([lstm_output.squeeze(1), context], dim=1)
        
        # Project to vocabulary size
        output = self.out(concat_output)
        
        return output, hidden, cell, attention_weights


# %% [code] {"execution":{"iopub.status.busy":"2025-07-17T09:39:26.750131Z","iopub.execute_input":"2025-07-17T09:39:26.750421Z","iopub.status.idle":"2025-07-17T09:39:26.776220Z","shell.execute_reply.started":"2025-07-17T09:39:26.750404Z","shell.execute_reply":"2025-07-17T09:39:26.775555Z"}}
import numpy as np
class Data2VisModel(nn.Module):
    """
    Main Data2Vis model implementing the sequence-to-sequence architecture
    with attention mechanism as described in the paper.
    """
    
    def __init__(self, source_vocab_size: int, target_vocab_size: int,
                 embedding_dim: int = 256, hidden_size: int = 512,
                 num_layers: int = 2, dropout_rate: float = 0.5):
        super(Data2VisModel, self).__init__()
        
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        # Encoder (bidirectional, so encoder hidden size is hidden_size * 2)
        self.encoder = EncoderLSTM(
            source_vocab_size, embedding_dim, hidden_size, num_layers, dropout_rate
        )
        
        # Decoder
        self.decoder = Decoder(
            target_vocab_size, embedding_dim, hidden_size, num_layers, dropout_rate
        )
        
        # Linear layer to project encoder hidden state to decoder hidden state
        self.hidden_projection = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, source, target, source_lengths, target_lengths):
        batch_size = source.size(0)
        max_target_length = target.size(1)
        
        # Encode source sequence
        encoder_outputs, encoder_hidden = self.encoder(source, source_lengths)
        
        # Initialize decoder hidden state
        decoder_hidden = self.hidden_projection(encoder_hidden).unsqueeze(0)
        decoder_cell = torch.zeros_like(decoder_hidden)
        
        # Repeat for each layer
        decoder_hidden = decoder_hidden.repeat(self.num_layers, 1, 1)
        decoder_cell = decoder_cell.repeat(self.num_layers, 1, 1)
        
        # Create mask for encoder outputs
        encoder_mask = self.create_mask(source, source_lengths)
        
        # Decoder forward pass
        outputs = []
        decoder_input = target[:, 0:1]  # Start token
        
        for t in range(1, max_target_length):
            output, decoder_hidden, decoder_cell, attention_weights = self.decoder(
                decoder_input, decoder_hidden, decoder_cell, encoder_outputs, encoder_mask
            )
            outputs.append(output)
            
            # Teacher forcing: use target as next input
            decoder_input = target[:, t:t+1]
        
        # Stack outputs
        outputs = torch.stack(outputs, dim=1)
        
        return outputs
    
    def create_mask(self, source, source_lengths):
        """
        Create mask for encoder outputs to ignore padding tokens.
        """
        batch_size = source.size(0)
        max_length = source.size(1)
        source_lengths = source_lengths.to(source.device)

        mask = torch.arange(max_length, device=source.device).expand(
            batch_size, max_length
        ) < source_lengths.unsqueeze(1)
        
        return mask
    
    def beam_search_decode(self, source, source_lengths, preprocessor,beam_width=15, max_length=1000, device='cpu'):
        """
        Fixed beam search decoding implementation.
        """
        self.eval()
        
        with torch.no_grad():
            batch_size = source.size(0)
            
            # Encode source sequence
            encoder_outputs, encoder_hidden = self.encoder(source, source_lengths)
            
            # Initialize decoder hidden state
            decoder_hidden = self.hidden_projection(encoder_hidden).unsqueeze(0)
            decoder_cell = torch.zeros_like(decoder_hidden)
            
            # Repeat for each layer
            decoder_hidden = decoder_hidden.repeat(self.num_layers, 1, 1)
            decoder_cell = decoder_cell.repeat(self.num_layers, 1, 1)
            
            # Create mask for encoder outputs
            encoder_mask = self.create_mask(source, source_lengths)
            
            # Initialize beam
            start_token = preprocessor.target_char_to_idx['<START>']
            end_token = preprocessor.target_char_to_idx['<END>']
            
            # Beam search for each sample in the batch
            results = []
            
            for b in range(batch_size):
                # Extract single sample
                sample_encoder_outputs = encoder_outputs[b:b+1]
                sample_encoder_mask = encoder_mask[b:b+1]
                sample_decoder_hidden = decoder_hidden[:, b:b+1, :]
                sample_decoder_cell = decoder_cell[:, b:b+1, :]
                
                # Fixed: Initialize beams with proper structure
                beams = [([], 0.0, sample_decoder_hidden.clone(), sample_decoder_cell.clone())]
                completed_sequences = []
                
                for step in range(max_length):
                    if not beams:
                        break
                        
                    new_beams = []
                    
                    for seq, score, hidden, cell in beams:
                        # Check if sequence is complete
                        if seq and seq[-1] == end_token:
                            # Length normalization
                            normalized_score = score / len(seq) if len(seq) > 0 else score
                            completed_sequences.append((seq, normalized_score))
                            continue
                        # Prevent infinite loops
                        if len(seq) >= max_length - 1:
                            seq_with_end = seq + [end_token]
                            normalized_score = score / len(seq_with_end)
                            completed_sequences.append((seq_with_end, normalized_score))
                            continue
                        # Get current input token
                        current_input = torch.tensor(
                            [[start_token if not seq else seq[-1]]], 
                            device=device, dtype=torch.long
                        )
                        
                        try:
                            # Predict next token
                            output, new_hidden, new_cell, _ = self.decoder(
                                current_input, hidden, cell, sample_encoder_outputs, sample_encoder_mask
                            )
                            
                            # Get probabilities and add small epsilon for numerical stability
                            probs = F.softmax(output, dim=-1)
                            probs = probs + 1e-10  # Avoid log(0)
                            
                            # Get top beam_width candidates
                            top_probs, top_indices = torch.topk(probs, min(beam_width, probs.size(-1)))
                            
                            for i in range(top_probs.size(1)):
                                idx = top_indices[0, i].item()
                                prob = top_probs[0, i].item()
                                
                                # Avoid log(0) by ensuring prob > 0
                                if prob > 1e-10:
                                    new_score = score + np.log(prob)
                                    new_seq = seq + [idx]
                                    
                                    new_beams.append((new_seq, new_score, new_hidden.clone(), new_cell.clone()))
                        except Exception as e:
                            print(f"Error in beam search step {step}: {e}")
                            # Add current sequence to completed with penalty
                            if seq:
                                penalized_score = score - 10.0  # Penalty for error
                                completed_sequences.append((seq + [end_token], penalized_score))
                            continue
                    
                    # Keep only top beam_width beams
                    if new_beams:
                        new_beams.sort(key=lambda x: x[1], reverse=True)
                        beams = new_beams[:beam_width]
                    else:
                        break
                
                # Add remaining beams to completed sequences
                for seq, score, _, _ in beams:
                    if seq:  # Only add non-empty sequences
                        if seq[-1] != end_token:
                            seq = seq + [end_token]
                        normalized_score = score / len(seq) if len(seq) > 0 else score
                        completed_sequences.append((seq, score))
                
                # Convert sequences back to text
                sample_results = []
                # Sort by score and take top results
                completed_sequences.sort(key=lambda x: x[1], reverse=True)
                
                for seq, score in completed_sequences[:beam_width]:
                    if seq:  # Ensure sequence is not empty
                        text = preprocessor.sequence_to_text(seq, preprocessor.target_idx_to_char)
                        if text and text != "":  # Only add non-empty text
                            sample_results.append(text)
                
                # If no valid results, add a default empty result
                if not sample_results:
                    sample_results = ["{}"]
                
                results.append(sample_results)
                
            return results



# %% [code] {"execution":{"iopub.status.busy":"2025-07-17T09:39:26.777154Z","iopub.execute_input":"2025-07-17T09:39:26.777424Z","iopub.status.idle":"2025-07-17T09:39:26.812723Z","shell.execute_reply.started":"2025-07-17T09:39:26.777400Z","shell.execute_reply":"2025-07-17T09:39:26.812002Z"}}
from tqdm import tqdm
import pickle
class Data2VisTrainer:
    """
    Main trainer class that handles the complete training pipeline.
    """
    
    def __init__(self, model_dir: str = "models"):
        
        self.model_dir = model_dir
        self.preprocessor = DataPreprocessor()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        print(f"Using device: {self.device}")
    
    def load_data(self) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
        """
        Load training, validation, and test data from JSON files.
        """
        def load_txt_file(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f]
    
        
        # Extract source and target sequences
        train_src = load_txt_file('train.sources')
        train_tgt = load_txt_file('train.targets')
        
        dev_src = load_txt_file('dev.sources')
        dev_tgt = load_txt_file('dev.targets')
        
        test_src = load_txt_file('test.sources')
        test_tgt = load_txt_file('test.targets')
        
        print(f"Loaded {len(train_src)} training examples")
        print(f"Loaded {len(dev_src)} validation examples")
        print(f"Loaded {len(test_src)} test examples")
        
        return train_src, train_tgt, dev_src, dev_tgt, test_src, test_tgt
    
    def preprocess_data(self, train_src: List[str], train_tgt: List[str],dev_src: List[str], dev_tgt: List[str],max_source_length: int = 500, max_target_length: int = 500):
        """
        Preprocess the data for training.
        """
        # Transform field names for all datasets
        transformed_train_src = []
        for src in train_src:
            transformed, _ = self.preprocessor.transform_field_names(src)
            transformed_train_src.append(transformed)
        
        transformed_dev_src = []
        for src in dev_src:
            transformed, _ = self.preprocessor.transform_field_names(src)
            transformed_dev_src.append(transformed)
        
        # Build vocabularies
        self.preprocessor.build_vocabularies(transformed_train_src, train_tgt)
        
        # Convert to sequences
        train_src_sequences = []
        for src in transformed_train_src:
            seq = self.preprocessor.text_to_sequence(
                src, self.preprocessor.source_char_to_idx, max_source_length
            )
            train_src_sequences.append(seq)
        
        train_tgt_sequences = []
        for tgt in train_tgt:
            seq = [self.preprocessor.target_char_to_idx['<START>']]
            seq.extend(self.preprocessor.text_to_sequence(
                tgt, self.preprocessor.target_char_to_idx, max_target_length-2
            ))
            seq.append(self.preprocessor.target_char_to_idx['<END>'])
            train_tgt_sequences.append(seq)
        
        # Do the same for validation data
        dev_src_sequences = []
        for src in transformed_dev_src:
            seq = self.preprocessor.text_to_sequence(
                src, self.preprocessor.source_char_to_idx, max_source_length
            )
            dev_src_sequences.append(seq)
        
        dev_tgt_sequences = []
        for tgt in dev_tgt:
            seq = [self.preprocessor.target_char_to_idx['<START>']]
            seq.extend(self.preprocessor.text_to_sequence(
                tgt, self.preprocessor.target_char_to_idx, max_target_length-2
            ))
            seq.append(self.preprocessor.target_char_to_idx['<END>'])
            dev_tgt_sequences.append(seq)
        
        return (train_src_sequences, train_tgt_sequences, 
                dev_src_sequences, dev_tgt_sequences)
    
    def train_model(self,num_steps=20000,  batch_size: int = 32, learning_rate: float = 0.0001):
        """
        Complete training pipeline.
        """
        # Load data
        train_src, train_tgt, dev_src, dev_tgt, test_src, test_tgt = self.load_data()
        
        # Preprocess data
        (train_src_seq, train_tgt_seq, dev_src_seq, dev_tgt_seq) = self.preprocess_data(
            train_src, train_tgt, dev_src, dev_tgt
        )
        
        # Create datasets
        train_dataset = Data2VisDataset(train_src_seq, train_tgt_seq,self.preprocessor)
        dev_dataset = Data2VisDataset(dev_src_seq, dev_tgt_seq,self.preprocessor)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )
        dev_loader = DataLoader(
            dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )
        
        # Build model
        self.model = Data2VisModel(
            source_vocab_size=len(self.preprocessor.source_char_to_idx),
            target_vocab_size=len(self.preprocessor.target_char_to_idx)
        ).to(self.device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=1000, verbose=True
        )
        print("Model built successfully!")
        print(f"Source vocab size: {len(self.preprocessor.source_char_to_idx)}")
        print(f"Target vocab size: {len(self.preprocessor.target_char_to_idx)}")
        
        # Training loop
        train_losses = []
        val_losses = []
        
        step = 0
        epoch = 0


        while step < num_steps:
            epoch += 1

            print(f"\nEpoch {epoch}")
            # Training
            self.model.train()
            train_loss_sum = 0.0
            train_batches = 0
            # Use tqdm for progress bar

            train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
            for batch in train_pbar:
                if step >= num_steps:
                    break
                source = batch['source'].to(self.device)
                target = batch['target'].to(self.device)
                source_lengths = batch['source_lengths'].to(self.device)
                target_lengths = batch['target_lengths'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(source, target, source_lengths, target_lengths)
                
                # Compute loss
                # Reshape for loss computation
                outputs = outputs.contiguous().view(-1, outputs.size(-1))
                target_shifted = target[:, 1:].contiguous().view(-1)
                
                loss = criterion(outputs, target_shifted)
                
                # Backward pass
                loss.backward()                    
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss_sum += loss.item()
                train_batches += 1
                
                step += 1

                # Update progress bar
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'step': f'{step}/{num_steps}'})



                if step % 100 == 0:
                    torch.cuda.empty_cache()  # Clear GPU memory periodically
                    print(f"Step {step}/{num_steps} - Loss: {loss.item():.4f}")
                if step % 1000 == 0:
                    checkpoint = {
                        'step': step,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item()
                    }
                    torch.save(checkpoint, f'checkpoint_step_{step}.pth')
            
            # Validation at end of each epoch
            if step < num_steps:  # Only validate if we haven't finished training
                self.model.eval()
                val_loss = 0.0
                val_batches = 0
                
                val_pbar = tqdm(dev_loader, desc=f"Validation Epoch {epoch}")
                
                with torch.no_grad():
                    for batch in val_pbar:
                        source = batch['source'].to(self.device)
                        target = batch['target'].to(self.device)
                        source_lengths = batch['source_lengths'].to(self.device)
                        target_lengths = batch['target_lengths'].to(self.device)
                        
                        outputs = self.model(source, target, source_lengths, target_lengths)
                        
                        outputs = outputs.contiguous().view(-1, outputs.size(-1))
                        target_shifted = target[:, 1:].contiguous().view(-1)
                        
                        loss = criterion(outputs, target_shifted)
                        val_loss += loss.item()
                        val_batches += 1
                        
                        val_pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
                
                # Calculate average losses for this epoch
                avg_train_loss = train_loss_sum / train_batches if train_batches > 0 else 0
                avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
                
                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)
                
                print(f"\nEpoch {epoch} Summary:")
                print(f"  Steps completed: {step}/{num_steps}")
                print(f"  Train Loss: {avg_train_loss:.4f}")
                print(f"  Val Loss: {avg_val_loss:.4f}")
        scheduler.step(avg_val_loss)
        # Save model
        self.save_model()
        
        return {'train_loss': train_losses, 'val_loss': val_losses}
    
    def save_model(self):
        """
        Save the trained model and preprocessor.
        """
        # Save model state dict
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, 'data2vis_model.pth'))
        
        # Save preprocessor
        with open(os.path.join(self.model_dir, 'preprocessor.pkl'), 'wb') as f:
            pickle.dump(self.preprocessor, f)
        
        # Save model configuration
        config = {
            'source_vocab_size': len(self.preprocessor.source_char_to_idx),
            'target_vocab_size': len(self.preprocessor.target_char_to_idx),
            'embedding_dim': self.model.embedding_dim,
            'hidden_size': self.model.hidden_size,
            'num_layers': self.model.num_layers,
            'dropout_rate': self.model.dropout_rate
        }
        
        with open(os.path.join(self.model_dir, 'model_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Model saved to {self.model_dir}")
    
    def load_model(self):
        """
        Load a trained model and preprocessor.
        """
        # Load preprocessor
        with open(os.path.join(self.model_dir, 'preprocessor.pkl'), 'rb') as f:
            self.preprocessor = pickle.load(f)
        
        # Load model configuration
        with open(os.path.join(self.model_dir, 'model_config.json'), 'r') as f:
            config = json.load(f)
        
        # Rebuild model
        self.model = Data2VisModel(**config).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(torch.load(
            os.path.join(self.model_dir, 'data2vis_model.pth'),
            map_location=self.device
        ))
        
        print("Model loaded successfully!")
    
    def generate_visualization(self, dataset_list: List[dict], beam_width: int = 15) -> List[dict]:
        """
        Generate visualization specifications for a list of dataset entries.
        Each record is encoded and passed individually, but combined data is injected into all specs.

        Args:
            dataset_list (List[dict]): List of input data rows (as dicts)
            beam_width (int): Beam width for decoding

        Returns:
            List[dict]: Combined visualization specs (1 per input, each referencing full dataset)
        """
        if self.model is None:
            self.load_model()

        if not dataset_list:
            print("Empty dataset list provided.")
            return []

        all_specs = []
        full_data_block = {"values": dataset_list}

        for i, record in enumerate(dataset_list):
            try:
                # Serialize record
                record_str = json.dumps(record)

                # Transform field names for generalization
                transformed_input, field_mapping = self.preprocessor.transform_field_names(record_str)

                # Tokenize input
                input_seq = self.preprocessor.text_to_sequence(
                    transformed_input, self.preprocessor.source_char_to_idx, max_length=500
                )

                source_tensor = torch.tensor([input_seq], dtype=torch.long).to(self.device)
                source_lengths = torch.tensor([len(input_seq)], dtype=torch.long).to(self.device)

                # Generate beam search outputs
                decoded_outputs = self.model.beam_search_decode(
                    source_tensor, source_lengths, self.preprocessor,
                    beam_width=beam_width, device=self.device
                )

                # Decode and attach full dataset
                for decoded_str in decoded_outputs[0]:
                    try:
                        restored_str = self.preprocessor.reverse_transform_field_names(decoded_str, field_mapping)
                        spec_dict = json.loads(restored_str)
                        spec_dict["data"] = full_data_block  # Attach full dataset
                        all_specs.append(spec_dict)
                    except Exception as e:
                        print(f"[Spec {i}] Failed to parse decoded string: {e}")
                        continue

            except Exception as e:
                print(f"[Record {i}] Failed: {e}")
                continue

        return all_specs




# %% [code]
import os
import matplotlib.pyplot as plt
def main():
    """
    Main function to demonstrate training and inference.
    """
    # Initialize trainer
    trainer = Data2VisTrainer()
    
    """ # Train model
    print("Starting training...")
    history = trainer.train_model(num_steps=20000, batch_size=32)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()  """
    
    # Example inference
    print("\nTesting inference...")
    
    # Load and preprocess progression.json
    try:
        with open("progression.json", "r") as f:
            dataset_list = json.load(f)  # This is a list of dicts
            visualizations = trainer.generate_visualization(dataset_list, beam_width=15)
            with open("combined_visualizations.json", "w") as f1:
                json.dump(visualizations, f1, indent=2)
    except FileNotFoundError:

        print("progression.json file not found. Please ensure the file exists.")

    except json.JSONDecodeError as e:

        print(f"Error parsing progression.json: {e}")

    except Exception as e:

        print(f"Unexpected error: {e}")
    
    
if __name__ == "__main__":
    main()
