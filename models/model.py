"""
Model architecture for the PeptideAI project.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Batch

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


class SequenceEmbedding(nn.Module):
    """
    Embedding layer for peptide sequences.
    """
    def __init__(self, embedding_dim=SEQ_EMBEDDING_DIM):
        """
        Initialize the embedding layer.
        
        Args:
            embedding_dim (int): Dimension of the embeddings.
        """
        super(SequenceEmbedding, self).__init__()
        
        # Amino acid embedding
        self.aa_embedding = nn.Embedding(len(AMINO_ACIDS), embedding_dim)
        
        # Position embedding
        self.position_embedding = nn.Embedding(SEQ_MAX_LENGTH, embedding_dim)
    
    def forward(self, sequences):
        """
        Forward pass.
        
        Args:
            sequences (list): List of peptide sequences.
            
        Returns:
            torch.Tensor: Embedded sequences.
        """
        # Convert sequences to indices
        batch_size = len(sequences)
        seq_lengths = [len(seq) for seq in sequences]
        max_length = max(seq_lengths)
        
        # Create tensor of amino acid indices
        aa_indices = torch.zeros(batch_size, max_length, dtype=torch.long, device=self.aa_embedding.weight.device)
        for i, seq in enumerate(sequences):
            for j, aa in enumerate(seq):
                aa_indices[i, j] = AMINO_ACIDS.index(aa)
        
        # Create tensor of position indices
        position_indices = torch.arange(max_length, device=self.position_embedding.weight.device).unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        aa_embeddings = self.aa_embedding(aa_indices)
        position_embeddings = self.position_embedding(position_indices)
        
        # Combine embeddings
        embeddings = aa_embeddings + position_embeddings
        
        # Create mask for padding
        mask = torch.zeros(batch_size, max_length, device=embeddings.device, dtype=torch.bool)
        for i, length in enumerate(seq_lengths):
            mask[i, length:] = True
        
        return embeddings, mask


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for peptide sequences.
    """
    def __init__(self, embedding_dim=SEQ_EMBEDDING_DIM, num_heads=SEQ_NUM_HEADS, num_layers=SEQ_NUM_LAYERS, dropout=SEQ_DROPOUT):
        """
        Initialize the transformer encoder.
        
        Args:
            embedding_dim (int): Dimension of the embeddings.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
            dropout (float): Dropout rate.
        """
        super(TransformerEncoder, self).__init__()
        
        # Embedding layer
        self.embedding = SequenceEmbedding(embedding_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, sequences):
        """
        Forward pass.
        
        Args:
            sequences (list): List of peptide sequences.
            
        Returns:
            dict: Dictionary with sequence representations and attention weights.
        """
        # Get embeddings and mask
        embeddings, mask = self.embedding(sequences)
        
        # Apply layer normalization
        embeddings = self.layer_norm(embeddings)
        
        # Apply transformer encoder
        # Convert mask to attention mask (1 for padding, 0 for non-padding)
        attention_mask = mask.float() * -1e9
        attention_mask = attention_mask.masked_fill(mask == 0, 0)
        
        # Forward pass through transformer
        sequence_output = self.transformer_encoder(embeddings, src_key_padding_mask=mask)
        
        # Get sequence representation (average of non-padding tokens)
        mask_expanded = mask.unsqueeze(-1).expand_as(sequence_output)
        sequence_output_masked = sequence_output.masked_fill(mask_expanded, 0)
        seq_lengths = torch.sum(~mask, dim=1).unsqueeze(-1)
        sequence_repr = torch.sum(sequence_output_masked, dim=1) / seq_lengths
        
        return {
            'sequence_output': sequence_output,
            'sequence_repr': sequence_repr,
            'attention_mask': mask
        }


class GraphEncoder(nn.Module):
    """
    Graph encoder for peptide graphs using PyTorch Geometric.
    """
    def __init__(self, input_dim=3, hidden_dim=GNN_HIDDEN_DIM, num_layers=GNN_NUM_LAYERS, dropout=GNN_DROPOUT):
        """
        Initialize the graph encoder.
        
        Args:
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Dimension of the hidden layers.
            num_layers (int): Number of graph convolutional layers.
            dropout (float): Dropout rate.
        """
        super(GraphEncoder, self).__init__()
        
        # Node embedding
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Graph convolutional layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gnn_layers.append(GATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // 4,  # Divide by 4 because we have 4 heads and concat=True
                heads=4,
                dropout=dropout,
                concat=True
            ))
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, graph):
        """
        Forward pass.
        
        Args:
            graph (torch_geometric.data.Data or list): Graph representation of the peptides.
            
        Returns:
            dict: Dictionary with graph representations and attention weights.
        """
        # Ensure graph is on the same device as the model
        device = next(self.parameters()).device
        
        # Handle case where graph is a list
        if isinstance(graph, list):
            print(f"Converting list of graphs to a single graph...")
            # If it's a list with a single graph, use that graph
            if len(graph) == 1:
                graph = graph[0]
            else:
                # If it's a list with multiple graphs, create a batch
                from torch_geometric.data import Batch
                try:
                    graph = Batch.from_data_list(graph)
                except Exception as e:
                    print(f"Error creating batch from list: {e}")
                    # Fallback to using the first graph
                    print(f"Using first graph as fallback")
                    graph = graph[0]
        
        # Check if graph has the necessary attributes
        if not hasattr(graph, 'x'):
            print(f"Graph does not have 'x' attribute. Creating a default graph...")
            # Create a default graph with basic features
            import torch
            from torch_geometric.data import Data
            
            # Create a simple chain graph with default features
            num_nodes = 10  # Default number of nodes
            edge_index = torch.tensor([[i, i+1] for i in range(num_nodes-1)], dtype=torch.long).t().contiguous()
            x = torch.ones((num_nodes, 3), dtype=torch.float)  # Default features
            
            # Create a batch index for global pooling
            batch = torch.zeros(num_nodes, dtype=torch.long)
            
            # Create the graph
            graph = Data(x=x, edge_index=edge_index, batch=batch)
            
            # Move to the correct device
            graph = graph.to(device)
        elif graph.x.device != device:
            graph = graph.to(device)
            
        # Get node features
        x = graph.x  # Shape: [num_nodes, 3]
        
        # Apply node embedding
        node_features = self.node_embedding(x)  # Shape: [num_nodes, hidden_dim]
        
        # Apply graph convolutional layers
        attention_weights = []
        for i, (gnn_layer, layer_norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            # Apply GNN layer
            node_features, attention = gnn_layer(node_features, graph.edge_index, return_attention_weights=True)
            
            # Apply layer normalization
            node_features = layer_norm(node_features)
            
            # Apply dropout
            node_features = self.dropout(node_features)
            
            # Store attention weights
            attention_weights.append(attention)
        
        # Get graph representation (average of node features)
        graph_repr = global_mean_pool(node_features, graph.batch)
        
        return {
            'node_features': node_features,
            'graph_repr': graph_repr,
            'attention_weights': attention_weights
        }


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention layer for sequence and graph representations.
    """
    def __init__(self, embedding_dim=SEQ_EMBEDDING_DIM, num_heads=CROSS_MODAL_HEADS, dropout=CROSS_MODAL_DROPOUT):
        """
        Initialize the cross-modal attention layer.
        
        Args:
            embedding_dim (int): Dimension of the embeddings.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super(CrossModalAttention, self).__init__()
        
        # Multi-head attention
        # IMPORTANT: Use batch_first=True for simplicity
        self.seq_to_graph_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.graph_to_seq_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Dimension alignment layer
        # This ensures that graph and sequence representations have the same dimension
        # before concatenation
        self.align_graph = nn.Linear(embedding_dim, embedding_dim)
        
        # Layer normalization
        self.seq_norm1 = nn.LayerNorm(embedding_dim)
        self.seq_norm2 = nn.LayerNorm(embedding_dim)
        self.graph_norm1 = nn.LayerNorm(embedding_dim)
        self.graph_norm2 = nn.LayerNorm(embedding_dim)
        
        # Feed-forward networks
        self.seq_ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout)
        )
        self.graph_ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, sequence_output, graph_output, sequence_mask=None):
        """
        Forward pass.
        
        Args:
            sequence_output (torch.Tensor): Sequence output from the transformer encoder.
            graph_output (torch.Tensor): Graph output from the graph encoder.
            sequence_mask (torch.Tensor, optional): Mask for padding in sequences.
            
        Returns:
            dict: Dictionary with fused representations and attention weights.
        """
        # Get dimensions
        batch_size, seq_length, embedding_dim = sequence_output.shape
        
        # Get number of heads and head dimension
        num_heads = self.seq_to_graph_attention.num_heads
        head_dim = embedding_dim // num_heads
        
        # Ensure embedding_dim is divisible by num_heads
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"
            )
        
        # Ensure graph_output has the right shape
        if len(graph_output.shape) == 2:
            # If graph_output is [batch_size, embedding_dim]
            graph_output_expanded = graph_output.unsqueeze(1).expand(-1, seq_length, -1)
        else:
            # If graph_output already has a sequence dimension
            graph_output_expanded = graph_output
            
        # Ensure dimensions match
        if graph_output_expanded.shape[1] != seq_length:
            # Resize to match sequence length
            graph_output_expanded = graph_output_expanded[:, :seq_length, :]
            
        # Ensure the embedding dimension matches
        if graph_output_expanded.shape[2] != embedding_dim:
            raise ValueError(
                f"graph_output embedding dimension ({graph_output_expanded.shape[2]}) "
                f"must match sequence_output embedding dimension ({embedding_dim})"
            )
        
        # Debug shapes
        # print(f"sequence_output shape: {sequence_output.shape}")
        # print(f"graph_output_expanded shape: {graph_output_expanded.shape}")
        # print(f"embedding_dim: {embedding_dim}, num_heads: {num_heads}, head_dim: {head_dim}")
        
        # Sequence to graph attention
        try:
            # For batch_first=True, inputs should be [batch_size, seq_len, embed_dim]
            # No need to transpose
            query = graph_output_expanded
            key = sequence_output
            value = sequence_output
            
            # Prepare mask for batch_first=True
            # key_padding_mask should be (batch_size, src_len) where src_len is the sequence length
            transposed_mask = None
            if sequence_mask is not None:
                # Get the expected shape parameters
                bsz = batch_size
                src_len = seq_length
                
                print(f"Original sequence_mask shape: {sequence_mask.shape}")
                print(f"Expected mask shape: [{bsz}, {src_len}]")
                
                # Create a properly sized mask
                transposed_mask = torch.zeros(bsz, src_len, dtype=torch.bool, device=sequence_mask.device)
                
                # Fill with the original mask values as much as possible
                min_batch = min(sequence_mask.shape[0], bsz)
                min_seq = min(sequence_mask.shape[1] if sequence_mask.dim() > 1 else 1, src_len)
                
                for i in range(min_batch):
                    for j in range(min_seq):
                        if sequence_mask.dim() > 1:
                            transposed_mask[i, j] = sequence_mask[i, j]
                        else:
                            transposed_mask[i, j] = sequence_mask[i]
                
                print(f"Prepared mask shape: {transposed_mask.shape}")
            
            # Apply attention
            # CRITICAL FIX: For MultiheadAttention with batch_first=True,
            # key_padding_mask should be [batch_size, src_len]
            # Force the correct shape regardless of previous transformations
            if transposed_mask is not None:
                # Get the expected dimensions
                bsz = batch_size  # batch size
                src_len = seq_length  # sequence length
                
                # Always ensure mask has shape [batch_size, src_len]
                if transposed_mask.shape != (bsz, src_len):
                    print(f"CRITICAL FIX: Forcing key_padding_mask from {transposed_mask.shape} to {(bsz, src_len)}")
                    
                    # If dimensions are flipped, transpose
                    if transposed_mask.shape == (src_len, bsz):
                        transposed_mask = transposed_mask.transpose(0, 1)
                    # If total elements match but shape is wrong, reshape
                    elif transposed_mask.numel() == bsz * src_len:
                        transposed_mask = transposed_mask.reshape(bsz, src_len)
                    # Otherwise create a new mask with the correct shape
                    else:
                        # Create a new mask with the correct shape
                        new_mask = torch.zeros(bsz, src_len, device=transposed_mask.device, dtype=transposed_mask.dtype)
                        # Fill with values from the original mask as much as possible
                        min_batch = min(transposed_mask.shape[0], bsz)
                        min_seq = min(transposed_mask.shape[1] if transposed_mask.dim() > 1 else 1, src_len)
                        for i in range(min_batch):
                            for j in range(min_seq):
                                if transposed_mask.dim() > 1:
                                    new_mask[i, j] = transposed_mask[i, j]
                                else:
                                    new_mask[i, j] = transposed_mask[i]
                        transposed_mask = new_mask
                
                print(f"Final key_padding_mask shape: {transposed_mask.shape}, expected: {(bsz, src_len)}")
                
            seq_attn_output, seq_attn_weights = self.seq_to_graph_attention(
                query=query,
                key=key,
                value=value,
                key_padding_mask=transposed_mask
            )
            
            # No need to transpose back since batch_first=True
        except RuntimeError as e:
            # If there's a shape error, create new tensors with the correct dimensions
            print(f"Error in seq_to_graph_attention: {e}")
            print(f"Creating new tensors with correct dimensions instead of reshaping...")
            
            # Create new tensors with the correct dimensions
            expected_dim = num_heads * head_dim
            
            # Create new query tensor
            new_query = torch.zeros(batch_size, seq_length, expected_dim, device=graph_output_expanded.device)
            # Copy data from original tensor as much as possible
            min_batch = min(graph_output_expanded.shape[0], batch_size)
            min_seq = min(graph_output_expanded.shape[1], seq_length)
            min_feat = min(graph_output_expanded.shape[2], expected_dim)
            new_query[:min_batch, :min_seq, :min_feat] = graph_output_expanded[:min_batch, :min_seq, :min_feat]
            
            # Create new key/value tensors
            new_key = torch.zeros(batch_size, seq_length, expected_dim, device=sequence_output.device)
            new_value = torch.zeros(batch_size, seq_length, expected_dim, device=sequence_output.device)
            min_batch = min(sequence_output.shape[0], batch_size)
            min_seq = min(sequence_output.shape[1], seq_length)
            min_feat = min(sequence_output.shape[2], expected_dim)
            new_key[:min_batch, :min_seq, :min_feat] = sequence_output[:min_batch, :min_seq, :min_feat]
            new_value[:min_batch, :min_seq, :min_feat] = sequence_output[:min_batch, :min_seq, :min_feat]
            
            print(f"Created new tensors with shapes: query={new_query.shape}, key={new_key.shape}, value={new_value.shape}")
            
            # Try again with new tensors
            seq_attn_output, seq_attn_weights = self.seq_to_graph_attention(
                query=new_query,
                key=new_key,
                value=new_value,
                key_padding_mask=transposed_mask
            )
            
        # Debug shapes before addition
        print(f"graph_output_expanded shape: {graph_output_expanded.shape}")
        print(f"seq_attn_output shape: {seq_attn_output.shape}")
        
        # Ensure shapes match before addition
        if graph_output_expanded.shape[0] != seq_attn_output.shape[0]:
            print(f"Shape mismatch in batch dimension: {graph_output_expanded.shape[0]} vs {seq_attn_output.shape[0]}")
            
            # Create a new tensor with the correct batch size
            min_batch = min(graph_output_expanded.shape[0], seq_attn_output.shape[0])
            
            # Resize both tensors to the same batch size
            graph_output_expanded = graph_output_expanded[:min_batch]
            seq_attn_output = seq_attn_output[:min_batch]
            
            print(f"Resized to batch size {min_batch}")
            print(f"New graph_output_expanded shape: {graph_output_expanded.shape}")
            print(f"New seq_attn_output shape: {seq_attn_output.shape}")
            
        seq_attn_output = self.seq_norm1(graph_output_expanded + seq_attn_output)
        seq_output = self.seq_norm2(seq_attn_output + self.seq_ffn(seq_attn_output))
        
        # Graph to sequence attention
        try:
            # For batch_first=True, inputs should be [batch_size, seq_len, embed_dim]
            # No need to transpose
            query = sequence_output
            key = graph_output_expanded
            value = graph_output_expanded
            
            # Prepare mask for batch_first=True
            # key_padding_mask should be (batch_size, src_len) where src_len is the sequence length
            transposed_mask = None
            if sequence_mask is not None:
                # Get the expected shape parameters
                bsz = batch_size
                src_len = seq_length
                
                print(f"Graph-to-seq original sequence_mask shape: {sequence_mask.shape}")
                print(f"Expected mask shape: [{bsz}, {src_len}]")
                
                # Create a properly sized mask
                transposed_mask = torch.zeros(bsz, src_len, dtype=torch.bool, device=sequence_mask.device)
                
                # Fill with the original mask values as much as possible
                min_batch = min(sequence_mask.shape[0], bsz)
                min_seq = min(sequence_mask.shape[1] if sequence_mask.dim() > 1 else 1, src_len)
                
                for i in range(min_batch):
                    for j in range(min_seq):
                        if sequence_mask.dim() > 1:
                            transposed_mask[i, j] = sequence_mask[i, j]
                        else:
                            transposed_mask[i, j] = sequence_mask[i]
                
                print(f"Prepared mask shape: {transposed_mask.shape}")
            
            # Apply attention
            # CRITICAL FIX: For MultiheadAttention with batch_first=True,
            # key_padding_mask should be [batch_size, src_len]
            # Force the correct shape regardless of previous transformations
            if transposed_mask is not None:
                # Get the expected dimensions
                bsz = batch_size  # batch size
                src_len = seq_length  # sequence length
                
                # Always ensure mask has shape [batch_size, src_len]
                if transposed_mask.shape != (bsz, src_len):
                    print(f"CRITICAL FIX: Forcing key_padding_mask from {transposed_mask.shape} to {(bsz, src_len)}")
                    
                    # If dimensions are flipped, transpose
                    if transposed_mask.shape == (src_len, bsz):
                        transposed_mask = transposed_mask.transpose(0, 1)
                    # If total elements match but shape is wrong, reshape
                    elif transposed_mask.numel() == bsz * src_len:
                        transposed_mask = transposed_mask.reshape(bsz, src_len)
                    # Otherwise create a new mask with the correct shape
                    else:
                        # Create a new mask with the correct shape
                        new_mask = torch.zeros(bsz, src_len, device=transposed_mask.device, dtype=transposed_mask.dtype)
                        # Fill with values from the original mask as much as possible
                        min_batch = min(transposed_mask.shape[0], bsz)
                        min_seq = min(transposed_mask.shape[1] if transposed_mask.dim() > 1 else 1, src_len)
                        for i in range(min_batch):
                            for j in range(min_seq):
                                if transposed_mask.dim() > 1:
                                    new_mask[i, j] = transposed_mask[i, j]
                                else:
                                    new_mask[i, j] = transposed_mask[i]
                        transposed_mask = new_mask
                
                print(f"Final key_padding_mask shape: {transposed_mask.shape}, expected: {(bsz, src_len)}")
                
            graph_attn_output, graph_attn_weights = self.graph_to_seq_attention(
                query=query,
                key=key,
                value=value,
                key_padding_mask=transposed_mask
            )
            
            # No need to transpose back since batch_first=True
        except RuntimeError as e:
            # If there's a shape error, create new tensors with the correct dimensions
            print(f"Error in graph_to_seq_attention: {e}")
            print(f"Creating new tensors with correct dimensions instead of reshaping...")
            
            # Create new tensors with the correct dimensions
            expected_dim = num_heads * head_dim
            
            # Create new query tensor
            new_query = torch.zeros(batch_size, seq_length, expected_dim, device=sequence_output.device)
            # Copy data from original tensor as much as possible
            min_batch = min(sequence_output.shape[0], batch_size)
            min_seq = min(sequence_output.shape[1], seq_length)
            min_feat = min(sequence_output.shape[2], expected_dim)
            new_query[:min_batch, :min_seq, :min_feat] = sequence_output[:min_batch, :min_seq, :min_feat]
            
            # Create new key/value tensors
            new_key = torch.zeros(batch_size, seq_length, expected_dim, device=graph_output_expanded.device)
            new_value = torch.zeros(batch_size, seq_length, expected_dim, device=graph_output_expanded.device)
            min_batch = min(graph_output_expanded.shape[0], batch_size)
            min_seq = min(graph_output_expanded.shape[1], seq_length)
            min_feat = min(graph_output_expanded.shape[2], expected_dim)
            new_key[:min_batch, :min_seq, :min_feat] = graph_output_expanded[:min_batch, :min_seq, :min_feat]
            new_value[:min_batch, :min_seq, :min_feat] = graph_output_expanded[:min_batch, :min_seq, :min_feat]
            
            print(f"Created new tensors with shapes: query={new_query.shape}, key={new_key.shape}, value={new_value.shape}")
            
            # Try again with new tensors
            graph_attn_output, graph_attn_weights = self.graph_to_seq_attention(
                query=new_query,
                key=new_key,
                value=new_value,
                key_padding_mask=None
            )
            
        # Debug shapes before addition
        print(f"sequence_output shape: {sequence_output.shape}")
        print(f"graph_attn_output shape: {graph_attn_output.shape}")
        
        # Ensure shapes match before addition
        if sequence_output.shape[0] != graph_attn_output.shape[0]:
            print(f"Shape mismatch in batch dimension: {sequence_output.shape[0]} vs {graph_attn_output.shape[0]}")
            
            # Create a new tensor with the correct batch size
            min_batch = min(sequence_output.shape[0], graph_attn_output.shape[0])
            
            # Resize both tensors to the same batch size
            sequence_output = sequence_output[:min_batch]
            graph_attn_output = graph_attn_output[:min_batch]
            
            print(f"Resized to batch size {min_batch}")
            print(f"New sequence_output shape: {sequence_output.shape}")
            print(f"New graph_attn_output shape: {graph_attn_output.shape}")
            
        graph_attn_output = self.graph_norm1(sequence_output + graph_attn_output)
        graph_output = self.graph_norm2(graph_attn_output + self.graph_ffn(graph_attn_output))
        
        # Ensure seq_output and graph_output have the same batch size
        if seq_output.shape[0] != graph_output.shape[0]:
            print(f"Shape mismatch in seq_output and graph_output: {seq_output.shape[0]} vs {graph_output.shape[0]}")
            min_batch = min(seq_output.shape[0], graph_output.shape[0])
            seq_output = seq_output[:min_batch]
            graph_output = graph_output[:min_batch]
            print(f"Adjusted batch sizes to {min_batch}")
            print(f"New seq_output shape: {seq_output.shape}")
            print(f"New graph_output shape: {graph_output.shape}")
        
        # Fuse representations
        seq_repr = seq_output.mean(1)  # [B, D]
        graph_repr = graph_output.mean(1)  # [B, D]
        
        # Debug shapes
        print(f"seq_repr shape: {seq_repr.shape}")
        print(f"graph_repr shape: {graph_repr.shape}")
        
        # Check if batch sizes match
        if seq_repr.shape[0] != graph_repr.shape[0]:
            # Adjust batch size to match
            min_batch_size = min(seq_repr.shape[0], graph_repr.shape[0])
            seq_repr = seq_repr[:min_batch_size]
            graph_repr = graph_repr[:min_batch_size]
            print(f"Adjusted batch sizes to {min_batch_size}")
        
        # Align graph representation dimension with sequence representation
        aligned_graph_repr = self.align_graph(graph_repr)  # [B, D]
        print(f"aligned_graph_repr shape: {aligned_graph_repr.shape}")
        
        # Ensure dimensions match for concatenation
        if seq_repr.shape[-1] != aligned_graph_repr.shape[-1]:
            # If dimensions don't match, resize the smaller one to match the larger one
            if seq_repr.shape[-1] < aligned_graph_repr.shape[-1]:
                # Create a new alignment layer on-the-fly if needed
                temp_align = nn.Linear(seq_repr.shape[-1], aligned_graph_repr.shape[-1]).to(seq_repr.device)
                seq_repr = temp_align(seq_repr)
                print(f"Adjusted seq_repr dimension to {seq_repr.shape}")
            else:
                # Create a new alignment layer on-the-fly if needed
                temp_align = nn.Linear(aligned_graph_repr.shape[-1], seq_repr.shape[-1]).to(aligned_graph_repr.device)
                aligned_graph_repr = temp_align(aligned_graph_repr)
                print(f"Adjusted aligned_graph_repr dimension to {aligned_graph_repr.shape}")
        
        # Concatenate aligned representations
        try:
            fused_repr = torch.cat([seq_repr, aligned_graph_repr], dim=-1)  # [B, D + D]
            print(f"fused_repr shape: {fused_repr.shape}")
        except RuntimeError as e:
            print(f"Error in concatenation: {e}")
            print(f"seq_repr shape: {seq_repr.shape}")
            print(f"aligned_graph_repr shape: {aligned_graph_repr.shape}")
            
            # As a fallback, use only the sequence representation
            print("Using only sequence representation as fallback")
            fused_repr = torch.cat([seq_repr, seq_repr], dim=-1)
        
        return {
            'seq_output': seq_output,
            'graph_output': graph_output,
            'fused_repr': fused_repr,
            'seq_attn_weights': seq_attn_weights,
            'graph_attn_weights': graph_attn_weights
        }


class PredictionHead(nn.Module):
    """
    Prediction head for binding energy prediction.
    """
    def __init__(self, input_dim, hidden_dim=256, dropout=0.1):
        """
        Initialize the prediction head.
        
        Args:
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Dimension of the hidden layer.
            dropout (float): Dropout rate.
        """
        super(PredictionHead, self).__init__()
        
        # Prediction layers
        self.prediction_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, features):
        """
        Forward pass.
        
        Args:
            features (torch.Tensor): Input features.
            
        Returns:
            torch.Tensor: Predicted binding energy.
        """
        # Apply prediction layers
        binding_energy = self.prediction_layers(features).squeeze(-1)
        
        return binding_energy


class ResidueContributionModule(nn.Module):
    """
    Module for computing residue contributions to binding energy.
    """
    def __init__(self, embedding_dim=SEQ_EMBEDDING_DIM):
        """
        Initialize the residue contribution module.
        
        Args:
            embedding_dim (int): Dimension of the embeddings.
        """
        super(ResidueContributionModule, self).__init__()
        
        # Contribution layers
        self.contribution_layers = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, 1)
        )
    
    def forward(self, sequence_output, sequence_mask=None):
        """
        Forward pass.
        
        Args:
            sequence_output (torch.Tensor): Sequence output from the transformer encoder.
            sequence_mask (torch.Tensor, optional): Mask for padding in sequences.
            
        Returns:
            torch.Tensor: Residue contributions.
        """
        # Apply contribution layers
        contributions = self.contribution_layers(sequence_output).squeeze(-1)
        
        # Apply mask if provided
        if sequence_mask is not None:
            contributions = contributions.masked_fill(sequence_mask, 0)
        
        return contributions


class PeptideBindingModel(nn.Module):
    """
    Model for peptide binding energy prediction.
    """
    def __init__(self, embedding_dim=SEQ_EMBEDDING_DIM, num_heads=SEQ_NUM_HEADS, num_layers=SEQ_NUM_LAYERS, dropout=SEQ_DROPOUT):
        """
        Initialize the model.
        
        Args:
            embedding_dim (int): Dimension of the embeddings.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
            dropout (float): Dropout rate.
        """
        super(PeptideBindingModel, self).__init__()
        
        # Sequence encoder
        self.sequence_encoder = TransformerEncoder(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Graph encoder
        self.graph_encoder = GraphEncoder(
            input_dim=3,  # hydrophobicity, weight, charge
            hidden_dim=embedding_dim,
            num_layers=GNN_NUM_LAYERS,
            dropout=GNN_DROPOUT
        )
        
        # Cross-modal attention
        self.cross_modal_attention = CrossModalAttention(
            embedding_dim=embedding_dim,
            num_heads=CROSS_MODAL_HEADS,
            dropout=CROSS_MODAL_DROPOUT
        )
        
        # Prediction head
        self.prediction_head = PredictionHead(
            input_dim=embedding_dim * 2,  # concatenated sequence and graph representations
            hidden_dim=embedding_dim * 2,
            dropout=dropout
        )
        
        # Residue contribution module
        self.residue_contribution = ResidueContributionModule(
            embedding_dim=embedding_dim
        )
    
    def forward(self, sequences, graph=None):
        """
        Forward pass.
        
        Args:
            sequences (list): List of peptide sequences.
            graph (torch_geometric.data.Data, optional): Graph representation of the peptides.
            
        Returns:
            dict: Dictionary with model outputs.
        """
        # Encode sequences
        sequence_outputs = self.sequence_encoder(sequences)
        sequence_output = sequence_outputs['sequence_output']
        sequence_repr = sequence_outputs['sequence_repr']
        sequence_mask = sequence_outputs['attention_mask']
        
        # Encode graph
        if graph is None:
            # Create graph if not provided
            from utils.data_processing import create_batch_graphs
            graph = create_batch_graphs(sequences)
            
            # Move graph to the same device as the model
            graph = graph.to(next(self.parameters()).device)
        
        # Encode graph
        graph_outputs = self.graph_encoder(graph)
        graph_repr = graph_outputs['graph_repr']
        
        # Apply cross-modal attention
        cross_modal_outputs = self.cross_modal_attention(
            sequence_output=sequence_output,
            graph_output=graph_repr,
            sequence_mask=sequence_mask
        )
        fused_repr = cross_modal_outputs['fused_repr']
        
        # Predict binding energy
        binding_energy = self.prediction_head(fused_repr)
        
        # Ensure sequence_output has the same batch size as fused_repr
        if sequence_output.shape[0] != fused_repr.shape[0]:
            print(f"Shape mismatch in sequence_output and fused_repr: {sequence_output.shape[0]} vs {fused_repr.shape[0]}")
            min_batch = min(sequence_output.shape[0], fused_repr.shape[0])
            sequence_output = sequence_output[:min_batch]
            sequence_mask = sequence_mask[:min_batch] if sequence_mask is not None else None
            print(f"Adjusted sequence_output batch size to {min_batch}")
            print(f"New sequence_output shape: {sequence_output.shape}")
        
        # Compute residue contributions
        residue_contributions = self.residue_contribution(sequence_output, sequence_mask)
        
        # Ensure sequence_repr and graph_repr have the same batch size as fused_repr
        if sequence_repr.shape[0] != fused_repr.shape[0]:
            print(f"Shape mismatch in sequence_repr and fused_repr: {sequence_repr.shape[0]} vs {fused_repr.shape[0]}")
            min_batch = min(sequence_repr.shape[0], fused_repr.shape[0])
            sequence_repr = sequence_repr[:min_batch]
            print(f"Adjusted sequence_repr batch size to {min_batch}")
            print(f"New sequence_repr shape: {sequence_repr.shape}")
            
        if graph_repr.shape[0] != fused_repr.shape[0]:
            print(f"Shape mismatch in graph_repr and fused_repr: {graph_repr.shape[0]} vs {fused_repr.shape[0]}")
            min_batch = min(graph_repr.shape[0], fused_repr.shape[0])
            graph_repr = graph_repr[:min_batch]
            print(f"Adjusted graph_repr batch size to {min_batch}")
            print(f"New graph_repr shape: {graph_repr.shape}")
            
        return {
            'binding_energy': binding_energy,
            'residue_contributions': residue_contributions,
            'sequence_repr': sequence_repr,
            'graph_repr': graph_repr,
            'fused_repr': fused_repr,
            'sequence_output': sequence_output,
            'sequence_mask': sequence_mask,
            'cross_modal_outputs': cross_modal_outputs
        }
