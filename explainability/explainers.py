"""
Explainability methods for the PeptideAI project.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import (
    IntegratedGradients,
    DeepLift,
    GradientShap,
    Occlusion,
    FeaturePermutation,
    FeatureAblation
)
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import os
import random
from collections import defaultdict

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


class LocalExplainer:
    """
    Local explainer for peptide binding energy prediction.
    """
    def __init__(self, model):
        """
        Initialize the local explainer.
        
        Args:
            model (nn.Module): Model to explain.
        """
        self.model = model
        
        # Create a wrapper module for Captum
        self.wrapper_module = self._create_wrapper_module()
        
        # Initialize attribution methods
        self.integrated_gradients = IntegratedGradients(self.wrapper_module)
        self.deep_lift = DeepLift(self.wrapper_module)
        self.gradient_shap = GradientShap(self.wrapper_module)
        self.occlusion = Occlusion(self.wrapper_module)
    
    def _create_wrapper_module(self):
        """
        Create a wrapper module for Captum.
        
        Returns:
            nn.Module: Wrapper module.
        """
        # Create a wrapper module that inherits from nn.Module
        # This is necessary because Captum expects a module, not a function
        model = self.model
        
        class WrapperModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model
                self.sequences = None
                self.sequence_mask = None
                self.graph = None
            
            def set_inputs(self, sequences, sequence_mask, graph):
                self.sequences = sequences
                self.sequence_mask = sequence_mask
                self.graph = graph
            
            def forward(self, embeddings):
                # Get original sequences and masks
                sequences = self.sequences
                sequence_mask = self.sequence_mask
                
                # Create custom forward function for attribution methods
                batch_size, seq_length, embedding_dim = embeddings.shape
                
                # Replace sequence output with provided embeddings
                sequence_output = embeddings
                
                # Get sequence representation (average of non-padding tokens)
                mask_expanded = sequence_mask.unsqueeze(-1).expand_as(sequence_output)
                sequence_output_masked = sequence_output.masked_fill(mask_expanded, 0)
                seq_lengths = torch.sum(~sequence_mask, dim=1).unsqueeze(-1)
                sequence_repr = torch.sum(sequence_output_masked, dim=1) / seq_lengths
                
                # Encode graph directly using the model's graph encoder
                graph_outputs = self.model.graph_encoder(self.graph)
                graph_repr = graph_outputs['graph_repr']
                
                # Apply cross-modal attention
                # Ensure sequence_mask has the correct shape for key_padding_mask
                if hasattr(graph_repr, 'shape'):
                    if len(graph_repr.shape) == 2:
                        # For 2D graph_repr, expand the mask to match expected shape
                        batch_size = sequence_mask.size(0)
                        src_len = 50  # Expected source length
                        
                        # Create a properly sized mask with the batch size as the first dimension
                        expanded_mask = sequence_mask.new_zeros(batch_size, src_len)
                        
                        # Fill with the original mask values
                        for i in range(batch_size):
                            expanded_mask[i, :] = sequence_mask[i, 0]  # Broadcast the mask value
                        
                        # Create a new mask with the correct shape
                        bsz = batch_size
                        src_len = 50  # Expected source length
                        
                        new_mask = torch.zeros(bsz, src_len, dtype=torch.bool, device=expanded_mask.device)
                        
                        # Fill with values from the original mask as much as possible
                        min_batch = min(expanded_mask.shape[0], bsz)
                        min_seq = min(expanded_mask.shape[1] if expanded_mask.dim() > 1 else 1, src_len)
                        
                        for i in range(min_batch):
                            for j in range(min_seq):
                                if expanded_mask.dim() > 1:
                                    new_mask[i, j] = expanded_mask[i, j]
                                else:
                                    new_mask[i, j] = expanded_mask[i]
                        
                        expanded_mask = new_mask
                        
                        cross_modal_outputs = self.model.cross_modal_attention(
                            sequence_output=sequence_output,
                            graph_output=graph_repr,
                            sequence_mask=expanded_mask
                        )
                    else:
                        # If graph_repr is 3D, use the original mask
                        cross_modal_outputs = self.model.cross_modal_attention(
                            sequence_output=sequence_output,
                            graph_output=graph_repr,
                            sequence_mask=sequence_mask
                        )
                else:
                    # Fallback if graph_repr doesn't have a shape attribute
                    cross_modal_outputs = self.model.cross_modal_attention(
                        sequence_output=sequence_output,
                        graph_output=graph_repr,
                        sequence_mask=sequence_mask
                    )
                fused_repr = cross_modal_outputs['fused_repr']
                
                # Predict binding energy
                binding_energy = self.model.prediction_head(fused_repr)
                
                # CRITICAL FIX: Captum expects output with shape [batch_size, num_classes]
                # For regression, we need to reshape to [batch_size, 1]
                if binding_energy.dim() == 1:
                    binding_energy = binding_energy.unsqueeze(1)
                
                return binding_energy
        
        return WrapperModule()
    
    def forward_wrapper(self, embeddings):
        """
        Wrapper for model forward pass.
        
        Args:
            embeddings (torch.Tensor): Sequence embeddings.
            
        Returns:
            torch.Tensor: Model output with shape [batch_size, num_classes] for Captum.
        """
        # Get original sequences and masks
        sequences = self.sequences
        sequence_mask = self.sequence_mask
        
        # Create custom forward function for attribution methods
        batch_size, seq_length, embedding_dim = embeddings.shape
        
        # Replace sequence output with provided embeddings
        sequence_output = embeddings
        
        # Get sequence representation (average of non-padding tokens)
        mask_expanded = sequence_mask.unsqueeze(-1).expand_as(sequence_output)
        sequence_output_masked = sequence_output.masked_fill(mask_expanded, 0)
        seq_lengths = torch.sum(~sequence_mask, dim=1).unsqueeze(-1)
        sequence_repr = torch.sum(sequence_output_masked, dim=1) / seq_lengths
        
        # For PyTorch Geometric graphs, we don't need to set node features
        # as they're already part of the graph. We'll just use the graph as is.
        # The model's forward method will handle the graph properly.
        
        # Encode graph directly using the model's graph encoder
        graph_outputs = self.model.graph_encoder(self.graph)
        graph_repr = graph_outputs['graph_repr']
        
        # Apply cross-modal attention
        # Ensure sequence_mask has the correct shape for key_padding_mask
        # It should be (batch_size, src_len) where src_len is the length of the graph representation
        # The error suggests we need (batch_size, 50) but got (batch_size, 1)
        
        # Create a properly sized mask
        if hasattr(graph_repr, 'shape'):
            if len(graph_repr.shape) == 2:
                # For 2D graph_repr, expand the mask to match expected shape (batch_size, 50)
                # The error shows we need (19, 50) but got (50, 1), so we need to ensure correct dimensions
                batch_size = sequence_mask.size(0)
                src_len = 50  # Expected source length
                
                # Create a properly sized mask with the batch size as the first dimension
                expanded_mask = sequence_mask.new_zeros(batch_size, src_len)
                
                # Fill with the original mask values
                for i in range(batch_size):
                    expanded_mask[i, :] = sequence_mask[i, 0]  # Broadcast the mask value across all positions
                
                print(f"Original sequence_mask shape: {sequence_mask.shape}")
                print(f"Expanded mask shape: {expanded_mask.shape}")
                print(f"Batch size: {batch_size}")
                
                # Create a properly sized mask for batch_first=True
                bsz = batch_size
                src_len = 50  # Expected source length
                
                # Always create a new mask with the correct shape to avoid any dimension issues
                print(f"CRITICAL FIX: Creating new mask with shape [{bsz}, {src_len}] for batch_first=True")
                
                # Create a new mask with the correct shape
                new_mask = torch.zeros(bsz, src_len, dtype=torch.bool, device=expanded_mask.device)
                
                # Fill with values from the original mask as much as possible
                min_batch = min(expanded_mask.shape[0], bsz)
                min_seq = min(expanded_mask.shape[1] if expanded_mask.dim() > 1 else 1, src_len)
                
                for i in range(min_batch):
                    for j in range(min_seq):
                        if expanded_mask.dim() > 1:
                            new_mask[i, j] = expanded_mask[i, j]
                        else:
                            new_mask[i, j] = expanded_mask[i]
                
                expanded_mask = new_mask
                
                print(f"Final key_padding_mask shape: {expanded_mask.shape}, expected: {(bsz, src_len)}")
                
                cross_modal_outputs = self.model.cross_modal_attention(
                    sequence_output=sequence_output,
                    graph_output=graph_repr,
                    sequence_mask=expanded_mask
                )
            else:
                # If graph_repr is 3D, use the original mask
                cross_modal_outputs = self.model.cross_modal_attention(
                    sequence_output=sequence_output,
                    graph_output=graph_repr,
                    sequence_mask=sequence_mask
                )
        else:
            # Fallback if graph_repr doesn't have a shape attribute
            cross_modal_outputs = self.model.cross_modal_attention(
                sequence_output=sequence_output,
                graph_output=graph_repr,
                sequence_mask=sequence_mask
            )
        fused_repr = cross_modal_outputs['fused_repr']
        
        # Predict binding energy
        binding_energy = self.model.prediction_head(fused_repr)
        
        # CRITICAL FIX: Captum expects output with shape [batch_size, num_classes]
        # For regression, we need to reshape to [batch_size, 1]
        if binding_energy.dim() == 1:
            binding_energy = binding_energy.unsqueeze(1)
            print(f"Reshaped binding_energy from 1D to 2D: {binding_energy.shape}")
        
        return binding_energy
    
    def explain(self, sequences, graph, method='integrated_gradients', baseline_type='zero'):
        """
        Explain model predictions.
        
        Args:
            sequences (list): List of peptide sequences.
            graph (dgl.DGLGraph): Graph representation of the peptides.
            method (str): Attribution method.
            baseline_type (str): Type of baseline to use.
            
        Returns:
            dict: Attribution results.
        """
        # Set sequences and graph for forward wrapper
        self.sequences = sequences
        self.graph = graph
        
        # Get model outputs
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(sequences, graph)
            sequence_output = outputs['sequence_output']
            self.sequence_mask = outputs['sequence_mask']
            predicted_energies = outputs['binding_energy']
        
        # Set inputs for the wrapper module
        self.wrapper_module.set_inputs(sequences, self.sequence_mask, graph)
        
        # Create baseline
        if baseline_type == 'zero':
            baseline = torch.zeros_like(sequence_output)
        elif baseline_type == 'random':
            baseline = torch.randn_like(sequence_output) * 0.1
        else:
            raise ValueError(f"Unknown baseline type: {baseline_type}")
        
        # Compute attributions
        if method == 'integrated_gradients':
            attributions = self.integrated_gradients.attribute(
                sequence_output,
                baselines=baseline,
                target=0,  # Target is always 0 for regression
                n_steps=50
            )
        elif method == 'deep_lift':
            try:
                attributions = self.deep_lift.attribute(
                    sequence_output,
                    baselines=baseline,
                    target=0
                )
            except Exception as e:
                print(f"Error with DeepLift: {e}")
                print("Falling back to IntegratedGradients...")
                attributions = self.integrated_gradients.attribute(
                    sequence_output,
                    baselines=baseline,
                    target=0,
                    n_steps=50
                )
        elif method == 'gradient_shap':
            try:
                attributions = self.gradient_shap.attribute(
                    sequence_output,
                    baselines=baseline,
                    target=0,
                    n_samples=50
                )
            except Exception as e:
                print(f"Error with GradientShap: {e}")
                print("Falling back to IntegratedGradients...")
                attributions = self.integrated_gradients.attribute(
                    sequence_output,
                    baselines=baseline,
                    target=0,
                    n_steps=50
                )
        elif method == 'occlusion':
            try:
                attributions = self.occlusion.attribute(
                    sequence_output,
                    sliding_window_shapes=(1, sequence_output.shape[2]),
                    baselines=baseline,
                    target=0
                )
            except Exception as e:
                print(f"Error with Occlusion: {e}")
                print("Falling back to IntegratedGradients...")
                attributions = self.integrated_gradients.attribute(
                    sequence_output,
                    baselines=baseline,
                    target=0,
                    n_steps=50
                )
        else:
            raise ValueError(f"Unknown attribution method: {method}")
        
        # Compute residue attributions (sum over embedding dimension)
        residue_attributions = attributions.sum(dim=2)
        
        # Apply mask
        residue_attributions = residue_attributions.masked_fill(self.sequence_mask, 0)
        
        # Normalize attributions
        normalized_attributions = []
        for i, seq in enumerate(sequences):
            seq_attr = residue_attributions[i, :len(seq)]
            if seq_attr.abs().sum() > 0:
                seq_attr = seq_attr / seq_attr.abs().sum()
            normalized_attributions.append(seq_attr)
        
        normalized_attributions = torch.stack(
            [torch.cat([attr, torch.zeros(residue_attributions.shape[1] - len(attr), device=attr.device)]) 
             for attr in normalized_attributions]
        )
        
        return {
            'attributions': attributions,
            'residue_attributions': normalized_attributions,
            'predicted_energies': predicted_energies,
            'method': method
        }
    
    def visualize(self, attribution_results, sequence_idx=0, save_path=None):
        """
        Visualize attributions for a specific sequence.
        
        Args:
            attribution_results (dict): Attribution results.
            sequence_idx (int): Index of the sequence to visualize.
            save_path (str, optional): Path to save the visualization.
            
        Returns:
            matplotlib.figure.Figure: Figure with attribution visualization.
        """
        # Get data
        attributions = attribution_results['attributions']
        residue_attributions = attribution_results['residue_attributions']
        predicted_energy = attribution_results['predicted_energies'][sequence_idx].item()
        method = attribution_results['method']
        
        # Get sequence
        sequence = self.sequences[sequence_idx]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot residue attributions - detach tensor before converting to numpy
        residue_attr = residue_attributions[sequence_idx, :len(sequence)].detach().cpu().numpy()
        ax.bar(range(len(sequence)), residue_attr)
        ax.set_xticks(range(len(sequence)))
        ax.set_xticklabels(list(sequence))
        ax.set_xlabel('Residue')
        ax.set_ylabel('Attribution')
        ax.set_title(f"Residue Attributions ({method})\nSequence: {sequence}, Predicted Energy: {predicted_energy:.2f}")
        ax.grid(True)
        
        # Add colorbar for attribution values
        sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=residue_attr.min(), vmax=residue_attr.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Attribution Value')
        
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class GlobalExplainer:
    """
    Global explainer for peptide binding energy prediction.
    """
    def __init__(self, model):
        """
        Initialize the global explainer.
        
        Args:
            model (nn.Module): Model to explain.
        """
        self.model = model
        
        # Initialize attribution methods
        self.feature_permutation = FeaturePermutation(self.forward_wrapper)
        self.feature_ablation = FeatureAblation(self.forward_wrapper)
    
    def forward_wrapper(self, embeddings):
        """
        Wrapper for model forward pass.
        
        Args:
            embeddings (torch.Tensor): Sequence embeddings.
            
        Returns:
            torch.Tensor: Model output.
        """
        # Similar to LocalExplainer.forward_wrapper
        sequences = self.sequences
        sequence_mask = self.sequence_mask
        
        batch_size, seq_length, embedding_dim = embeddings.shape
        
        sequence_output = embeddings
        
        mask_expanded = sequence_mask.unsqueeze(-1).expand_as(sequence_output)
        sequence_output_masked = sequence_output.masked_fill(mask_expanded, 0)
        seq_lengths = torch.sum(~sequence_mask, dim=1).unsqueeze(-1)
        sequence_repr = torch.sum(sequence_output_masked, dim=1) / seq_lengths
        
        # For PyTorch Geometric graphs, we don't need to set node features
        # as they're already part of the graph. We'll just use the graph as is.
        
        # Encode graph directly using the model's graph encoder
        graph_outputs = self.model.graph_encoder(self.graph)
        graph_repr = graph_outputs['graph_repr']
        
        # Apply cross-modal attention
        # Ensure sequence_mask has the correct shape for key_padding_mask
        # It should be (batch_size, src_len) where src_len is the length of the graph representation
        # The error suggests we need (batch_size, 50) but got (batch_size, 1)
        
        # Create a properly sized mask
        if hasattr(graph_repr, 'shape'):
            if len(graph_repr.shape) == 2:
                # For 2D graph_repr, expand the mask to match expected shape (batch_size, 50)
                # The error shows we need (19, 50) but got (50, 1), so we need to ensure correct dimensions
                batch_size = sequence_mask.size(0)
                src_len = 50  # Expected source length
                
                # Create a properly sized mask with the batch size as the first dimension
                expanded_mask = sequence_mask.new_zeros(batch_size, src_len)
                
                # Fill with the original mask values
                for i in range(batch_size):
                    expanded_mask[i, :] = sequence_mask[i, 0]  # Broadcast the mask value across all positions
                
                print(f"GlobalExplainer - Original sequence_mask shape: {sequence_mask.shape}")
                print(f"GlobalExplainer - Expanded mask shape: {expanded_mask.shape}")
                print(f"GlobalExplainer - Batch size: {batch_size}")
                
                # CRITICAL FIX: Force the correct shape regardless of previous transformations
                # Get the expected dimensions
                bsz = batch_size  # batch size
                src_len = 50  # sequence length
                
                # Always create a new mask with the correct shape to avoid any dimension issues
                print(f"GlobalExplainer - CRITICAL FIX: Creating new mask with shape [{bsz}, {src_len}] for batch_first=True")
                
                # Create a new mask with the correct shape
                new_mask = torch.zeros(bsz, src_len, device=expanded_mask.device, dtype=expanded_mask.dtype)
                
                # Fill with values from the original mask as much as possible
                min_batch = min(expanded_mask.shape[0], bsz)
                min_seq = min(expanded_mask.shape[1] if expanded_mask.dim() > 1 else 1, src_len)
                
                for i in range(min_batch):
                    for j in range(min_seq):
                        if expanded_mask.dim() > 1:
                            new_mask[i, j] = expanded_mask[i, j]
                        else:
                            new_mask[i, j] = expanded_mask[i]
                
                expanded_mask = new_mask
                
                print(f"GlobalExplainer - Final key_padding_mask shape: {expanded_mask.shape}, expected: {(bsz, src_len)}")
                
                cross_modal_outputs = self.model.cross_modal_attention(
                    sequence_output=sequence_output,
                    graph_output=graph_repr,
                    sequence_mask=expanded_mask
                )
            else:
                # If graph_repr is 3D, use the original mask
                cross_modal_outputs = self.model.cross_modal_attention(
                    sequence_output=sequence_output,
                    graph_output=graph_repr,
                    sequence_mask=sequence_mask
                )
        else:
            # Fallback if graph_repr doesn't have a shape attribute
            cross_modal_outputs = self.model.cross_modal_attention(
                sequence_output=sequence_output,
                graph_output=graph_repr,
                sequence_mask=sequence_mask
            )
        fused_repr = cross_modal_outputs['fused_repr']
        
        binding_energy = self.model.prediction_head(fused_repr)
        
        # CRITICAL FIX: Captum expects output with shape [batch_size, num_classes]
        # For regression, we need to reshape to [batch_size, 1]
        if binding_energy.dim() == 1:
            binding_energy = binding_energy.unsqueeze(1)
            print(f"GlobalExplainer - Reshaped binding_energy from 1D to 2D: {binding_energy.shape}")
        
        return binding_energy
    
    def explain(self, sequences, graph, method='permutation_importance'):
        """
        Explain model predictions globally.
        
        Args:
            sequences (list): List of peptide sequences.
            graph (dgl.DGLGraph): Graph representation of the peptides.
            method (str): Attribution method.
            
        Returns:
            dict: Attribution results.
        """
        # Set sequences and graph for forward wrapper
        self.sequences = sequences
        self.graph = graph
        
        # Get model outputs
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(sequences, graph)
            sequence_output = outputs['sequence_output']
            self.sequence_mask = outputs['sequence_mask']
            predicted_energies = outputs['binding_energy']
        
        # Compute feature importance
        if method == 'permutation_importance':
            # Compute permutation importance for each position
            importance_by_position = {}
            
            # Get baseline prediction
            baseline_pred = predicted_energies.detach().clone()
            
            # For each position, permute across sequences and compute importance
            max_length = sequence_output.shape[1]
            for pos in range(max_length):
                # Skip if all sequences are shorter than this position
                if all(len(seq) <= pos for seq in sequences):
                    continue
                
                # Create permuted sequences
                permuted_sequences = []
                for seq in sequences:
                    if len(seq) > pos:
                        # Randomly select another sequence with same length or longer
                        candidates = [s for s in sequences if len(s) > pos and s != seq]
                        if candidates:
                            other_seq = random.choice(candidates)
                            permuted_seq = seq[:pos] + other_seq[pos] + seq[pos+1:]
                        else:
                            permuted_seq = seq
                    else:
                        permuted_seq = seq
                    permuted_sequences.append(permuted_seq)
                
                # Get predictions for permuted sequences
                with torch.no_grad():
                    permuted_outputs = self.model(permuted_sequences, graph)
                    permuted_pred = permuted_outputs['binding_energy']
                
                # Compute importance as mean absolute difference
                importance = torch.abs(baseline_pred - permuted_pred).mean().item()
                
                # Store importance
                importance_by_position[pos] = importance
            
            # Compute amino acid importance
            importance_by_aa = {}
            
            # For each amino acid, permute and compute importance
            for aa in AMINO_ACIDS:
                # Create permuted sequences
                permuted_sequences = []
                for seq in sequences:
                    # Replace all occurrences of the amino acid with a random other amino acid
                    permuted_seq = ""
                    for c in seq:
                        if c == aa:
                            other_aa = random.choice([a for a in AMINO_ACIDS if a != aa])
                            permuted_seq += other_aa
                        else:
                            permuted_seq += c
                    permuted_sequences.append(permuted_seq)
                
                # Get predictions for permuted sequences
                with torch.no_grad():
                    permuted_outputs = self.model(permuted_sequences, graph)
                    permuted_pred = permuted_outputs['binding_energy']
                
                # Compute importance as mean absolute difference
                importance = torch.abs(baseline_pred - permuted_pred).mean().item()
                
                # Store importance
                importance_by_aa[aa] = importance
            
            return {
                'importance_by_position': importance_by_position,
                'importance_by_aa': importance_by_aa,
                'method': method
            }
        
        elif method == 'feature_ablation':
            # Use Captum's FeatureAblation
            attributions = self.feature_ablation.attribute(
                sequence_output,
                target=0,
                feature_mask=~self.sequence_mask.unsqueeze(-1).expand_as(sequence_output)
            )
            
            # Compute residue attributions (sum over embedding dimension)
            residue_attributions = attributions.sum(dim=2)
            
            # Apply mask
            residue_attributions = residue_attributions.masked_fill(self.sequence_mask, 0)
            
            # Compute importance by position
            importance_by_position = {}
            for pos in range(residue_attributions.shape[1]):
                # Skip if all sequences are masked at this position
                if self.sequence_mask[:, pos].all():
                    continue
                
                # Compute mean importance for this position
                importance = residue_attributions[:, pos].abs().mean().item()
                importance_by_position[pos] = importance
            
            # Compute importance by amino acid
            importance_by_aa = defaultdict(list)
            for i, seq in enumerate(sequences):
                for j, aa in enumerate(seq):
                    importance_by_aa[aa].append(residue_attributions[i, j].item())
            
            # Compute mean importance for each amino acid
            importance_by_aa = {aa: np.mean(np.abs(values)) for aa, values in importance_by_aa.items()}
            
            return {
                'importance_by_position': importance_by_position,
                'importance_by_aa': importance_by_aa,
                'method': method
            }
        
        else:
            raise ValueError(f"Unknown attribution method: {method}")
    
    def visualize(self, attribution_results, save_path=None):
        """
        Visualize global attributions.
        
        Args:
            attribution_results (dict): Attribution results.
            save_path (str, optional): Path to save the visualization.
            
        Returns:
            matplotlib.figure.Figure: Figure with attribution visualization.
        """
        # Get data
        importance_by_position = attribution_results['importance_by_position']
        importance_by_aa = attribution_results['importance_by_aa']
        method = attribution_results['method']
        
        # Create figure
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot importance by position
        positions = list(importance_by_position.keys())
        importances = list(importance_by_position.values())
        
        axs[0].bar(positions, importances)
        axs[0].set_xlabel('Position')
        axs[0].set_ylabel('Importance')
        axs[0].set_title(f"Position Importance ({method})")
        axs[0].grid(True)
        
        # Plot importance by amino acid
        aa_list = list(importance_by_aa.keys())
        aa_importances = list(importance_by_aa.values())
        
        # Sort by importance
        sorted_indices = np.argsort(aa_importances)[::-1]
        sorted_aa = [aa_list[i] for i in sorted_indices]
        sorted_importances = [aa_importances[i] for i in sorted_indices]
        
        axs[1].bar(sorted_aa, sorted_importances)
        axs[1].set_xlabel('Amino Acid')
        axs[1].set_ylabel('Importance')
        axs[1].set_title(f"Amino Acid Importance ({method})")
        axs[1].grid(True)
        
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class ThermodynamicExplainer:
    """
    Thermodynamic explainer for peptide binding energy prediction.
    """
    def __init__(self, model):
        """
        Initialize the thermodynamic explainer.
        
        Args:
            model (nn.Module): Model to explain.
        """
        self.model = model
    
    def explain(self, sequences, graph):
        """
        Generate thermodynamic maps for peptide sequences.
        
        Args:
            sequences (list): List of peptide sequences.
            graph (dgl.DGLGraph): Graph representation of the peptides.
            
        Returns:
            dict: Thermodynamic maps.
        """
        # Get model outputs
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(sequences, graph)
            sequence_output = outputs['sequence_output']
            sequence_mask = outputs['sequence_mask']
            predicted_energies = outputs['binding_energy']
            residue_contributions = outputs['residue_contributions']
        
        # Compute thermodynamic parameters
        thermo_maps = []
        
        for i, seq in enumerate(sequences):
            # Get residue contributions
            contrib = residue_contributions[i, :len(seq)].cpu().numpy()
            
            # Compute thermodynamic parameters
            enthalpy = contrib * predicted_energies[i].item()
            entropy = -contrib * np.log(np.abs(contrib) + 1e-10)
            free_energy = enthalpy - 298.15 * entropy  # G = H - TS, T = 298.15K (25°C)
            
            thermo_map = {
                'sequence': seq,
                'predicted_energy': predicted_energies[i].item(),
                'residue_contributions': contrib,
                'enthalpy': enthalpy,
                'entropy': entropy,
                'free_energy': free_energy
            }
            
            thermo_maps.append(thermo_map)
        
        return {
            'thermo_maps': thermo_maps
        }
    
    def visualize(self, thermo_results, sequence_idx=0, save_path=None):
        """
        Visualize thermodynamic map for a specific sequence.
        
        Args:
            thermo_results (dict): Thermodynamic results.
            sequence_idx (int): Index of the sequence to visualize.
            save_path (str, optional): Path to save the visualization.
            
        Returns:
            matplotlib.figure.Figure: Figure with thermodynamic visualization.
        """
        # Get data
        thermo_map = thermo_results['thermo_maps'][sequence_idx]
        sequence = thermo_map['sequence']
        predicted_energy = thermo_map['predicted_energy']
        residue_contributions = thermo_map['residue_contributions']
        enthalpy = thermo_map['enthalpy']
        entropy = thermo_map['entropy']
        free_energy = thermo_map['free_energy']
        
        # Create figure
        fig, axs = plt.subplots(3, 1, figsize=(12, 12))
        
        # Plot enthalpy
        axs[0].bar(range(len(sequence)), enthalpy)
        axs[0].set_xticks(range(len(sequence)))
        axs[0].set_xticklabels(list(sequence))
        axs[0].set_xlabel('Residue')
        axs[0].set_ylabel('Enthalpy (kcal/mol)')
        axs[0].set_title(f"Enthalpy Contribution\nSequence: {sequence}, Predicted Energy: {predicted_energy:.2f}")
        axs[0].grid(True)
        
        # Plot entropy
        axs[1].bar(range(len(sequence)), entropy)
        axs[1].set_xticks(range(len(sequence)))
        axs[1].set_xticklabels(list(sequence))
        axs[1].set_xlabel('Residue')
        axs[1].set_ylabel('Entropy (kcal/mol/K)')
        axs[1].set_title("Entropy Contribution")
        axs[1].grid(True)
        
        # Plot free energy
        axs[2].bar(range(len(sequence)), free_energy)
        axs[2].set_xticks(range(len(sequence)))
        axs[2].set_xticklabels(list(sequence))
        axs[2].set_xlabel('Residue')
        axs[2].set_ylabel('Free Energy (kcal/mol)')
        axs[2].set_title("Free Energy Contribution")
        axs[2].grid(True)
        
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class CounterfactualExplainer:
    """
    Counterfactual explainer for peptide binding energy prediction.
    """
    def __init__(self, model):
        """
        Initialize the counterfactual explainer.
        
        Args:
            model (nn.Module): Model to explain.
        """
        self.model = model
    
    def explain(self, sequence, graph, num_samples=COUNTERFACTUAL_NUM_SAMPLES, target_diff=2.0):
        """
        Generate counterfactual explanations for a peptide sequence.
        
        Args:
            sequence (str): Peptide sequence.
            graph (dgl.DGLGraph): Graph representation of the peptide.
            num_samples (int): Number of counterfactual samples to generate.
            target_diff (float): Target difference in binding energy.
            
        Returns:
            dict: Counterfactual explanations.
        """
        # Get model prediction for original sequence
        self.model.eval()
        with torch.no_grad():
            outputs = self.model([sequence], graph)
            original_energy = outputs['binding_energy'].item()
            residue_contributions = outputs['residue_contributions'][0, :len(sequence)].cpu().numpy()
        
        # Generate counterfactual sequences
        counterfactuals = []
        
        # Sort residues by contribution
        residue_indices = np.argsort(np.abs(residue_contributions))[::-1]
        
        # Generate counterfactuals by mutating high-contribution residues
        for _ in range(num_samples):
            # Choose number of residues to mutate (1-3)
            num_mutations = random.randint(1, min(3, len(sequence)))
            
            # Choose residues to mutate
            mutation_indices = residue_indices[:num_mutations]
            
            # Create counterfactual sequence
            cf_sequence = list(sequence)
            for idx in mutation_indices:
                # Choose a random amino acid different from the original
                original_aa = sequence[idx]
                new_aa = random.choice([aa for aa in AMINO_ACIDS if aa != original_aa])
                cf_sequence[idx] = new_aa
            
            cf_sequence = ''.join(cf_sequence)
            
            # Get prediction for counterfactual sequence
            with torch.no_grad():
                from utils.data_processing import create_batch_graphs
                cf_graph = create_batch_graphs([cf_sequence])
                
                # Get device from one of the tensors in the graph
                # PyTorch Geometric's Data objects don't have a direct device attribute
                if hasattr(graph, 'x') and hasattr(graph.x, 'device'):
                    device = graph.x.device
                elif hasattr(graph, 'edge_index') and hasattr(graph.edge_index, 'device'):
                    device = graph.edge_index.device
                else:
                    # Fallback to model's device
                    device = next(self.model.parameters()).device
                    
                cf_graph = cf_graph.to(device)
                
                cf_outputs = self.model([cf_sequence], cf_graph)
                cf_energy = cf_outputs['binding_energy'].item()
                cf_residue_contributions = cf_outputs['residue_contributions'][0, :len(cf_sequence)].cpu().numpy()
            
            # Check if counterfactual has significant difference
            energy_diff = cf_energy - original_energy
            
            if abs(energy_diff) >= target_diff:
                counterfactual = {
                    'sequence': cf_sequence,
                    'energy': cf_energy,
                    'energy_diff': energy_diff,
                    'mutations': [(idx, sequence[idx], cf_sequence[idx]) for idx in mutation_indices],
                    'residue_contributions': cf_residue_contributions
                }
                counterfactuals.append(counterfactual)
        
        # Sort counterfactuals by absolute energy difference
        counterfactuals.sort(key=lambda x: abs(x['energy_diff']), reverse=True)
        
        return {
            'original_sequence': sequence,
            'original_energy': original_energy,
            'original_contributions': residue_contributions,
            'counterfactuals': counterfactuals
        }
    
    def visualize(self, counterfactual_results, save_path=None):
        """
        Visualize counterfactual explanations.
        
        Args:
            counterfactual_results (dict): Counterfactual results.
            save_path (str, optional): Path to save the visualization.
            
        Returns:
            matplotlib.figure.Figure: Figure with counterfactual visualization.
        """
        # Get data
        original_sequence = counterfactual_results['original_sequence']
        original_energy = counterfactual_results['original_energy']
        original_contributions = counterfactual_results['original_contributions']
        counterfactuals = counterfactual_results['counterfactuals']
        
        # Check if counterfactuals were found
        if not counterfactuals:
            print("No counterfactuals found.")
            return None
        
        # Select top counterfactuals
        top_counterfactuals = counterfactuals[:min(3, len(counterfactuals))]
        
        # Create figure
        fig, axs = plt.subplots(len(top_counterfactuals) + 1, 1, figsize=(12, 4 * (len(top_counterfactuals) + 1)))
        
        # Plot original sequence
        axs[0].bar(range(len(original_sequence)), original_contributions)
        axs[0].set_xticks(range(len(original_sequence)))
        axs[0].set_xticklabels(list(original_sequence))
        axs[0].set_xlabel('Residue')
        axs[0].set_ylabel('Contribution')
        axs[0].set_title(f"Original Sequence: {original_sequence}\nBinding Energy: {original_energy:.2f}")
        axs[0].grid(True)
        
        # Plot counterfactuals
        for i, cf in enumerate(top_counterfactuals):
            # Get data
            cf_sequence = cf['sequence']
            cf_energy = cf['energy']
            cf_energy_diff = cf['energy_diff']
            cf_contributions = cf['residue_contributions']
            mutations = cf['mutations']
            
            # Plot contributions
            bars = axs[i+1].bar(range(len(cf_sequence)), cf_contributions)
            
            # Highlight mutated residues
            for idx, _, _ in mutations:
                bars[idx].set_color('red')
            
            axs[i+1].set_xticks(range(len(cf_sequence)))
            axs[i+1].set_xticklabels(list(cf_sequence))
            axs[i+1].set_xlabel('Residue')
            axs[i+1].set_ylabel('Contribution')
            
            # Create mutation string
            mutation_str = ', '.join([f"{orig_aa}{idx+1}{new_aa}" for idx, orig_aa, new_aa in mutations])
            
            axs[i+1].set_title(f"Counterfactual {i+1}: {cf_sequence}\nBinding Energy: {cf_energy:.2f} (Diff: {cf_energy_diff:+.2f})\nMutations: {mutation_str}")
            axs[i+1].grid(True)
        
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class ExplainabilityManager:
    """
    Manager for explainability methods.
    """
    def __init__(self, model):
        """
        Initialize the explainability manager.
        
        Args:
            model (nn.Module): Model to explain.
        """
        self.model = model
        self.local_explainer = LocalExplainer(model)
        self.global_explainer = GlobalExplainer(model)
        self.thermodynamic_explainer = ThermodynamicExplainer(model)
        self.counterfactual_explainer = CounterfactualExplainer(model)
    
    def explain_local(self, sequences, graph, method='integrated_gradients'):
        """
        Generate local explanations.
        
        Args:
            sequences (list): List of peptide sequences.
            graph (torch_geometric.data.Data or list): Graph representation of the peptides.
            method (str): Attribution method.
            
        Returns:
            dict: Local explanations.
        """
        # Check if graph is a list and convert to proper graph object if needed
        if isinstance(graph, list):
            print("Converting list to proper graph object...")
            from utils.data_processing import create_batch_graphs
            graph = create_batch_graphs(sequences)
            # Move graph to the same device as the model
            graph = graph.to(next(self.model.parameters()).device)
            print(f"Converted graph type: {type(graph)}")
        
        return self.local_explainer.explain(sequences, graph, method)
    
    def explain_global(self, sequences, graph, method='permutation_importance'):
        """
        Generate global explanations.
        
        Args:
            sequences (list): List of peptide sequences.
            graph (torch_geometric.data.Data or list): Graph representation of the peptides.
            method (str): Attribution method.
            
        Returns:
            dict: Global explanations.
        """
        # Check if graph is a list and convert to proper graph object if needed
        if isinstance(graph, list):
            print("Converting list to proper graph object...")
            from utils.data_processing import create_batch_graphs
            graph = create_batch_graphs(sequences)
            # Move graph to the same device as the model
            graph = graph.to(next(self.model.parameters()).device)
            print(f"Converted graph type: {type(graph)}")
        
        return self.global_explainer.explain(sequences, graph, method)
    
    def explain_thermodynamic(self, sequences, graph):
        """
        Generate thermodynamic explanations.
        
        Args:
            sequences (list): List of peptide sequences.
            graph (torch_geometric.data.Data or list): Graph representation of the peptides.
            
        Returns:
            dict: Thermodynamic explanations.
        """
        # Check if graph is a list and convert to proper graph object if needed
        if isinstance(graph, list):
            print("Converting list to proper graph object...")
            from utils.data_processing import create_batch_graphs
            graph = create_batch_graphs(sequences)
            # Move graph to the same device as the model
            graph = graph.to(next(self.model.parameters()).device)
            print(f"Converted graph type: {type(graph)}")
        
        return self.thermodynamic_explainer.explain(sequences, graph)
    
    def explain_counterfactual(self, sequence, graph):
        """
        Generate counterfactual explanations.
        
        Args:
            sequence (str): Peptide sequence.
            graph (torch_geometric.data.Data or list): Graph representation of the peptide.
            
        Returns:
            dict: Counterfactual explanations.
        """
        # Check if graph is a list and convert to proper graph object if needed
        if isinstance(graph, list):
            print("Converting list to proper graph object...")
            from utils.data_processing import create_batch_graphs
            graph = create_batch_graphs([sequence])
            # Move graph to the same device as the model
            graph = graph.to(next(self.model.parameters()).device)
            print(f"Converted graph type: {type(graph)}")
        
        return self.counterfactual_explainer.explain(sequence, graph)
    
    def visualize_local(self, attribution_results, save_path=None):
        """
        Visualize local explanations.
        
        Args:
            attribution_results (dict): Attribution results.
            save_path (str, optional): Path to save the visualization.
            
        Returns:
            matplotlib.figure.Figure: Figure with attribution visualization.
        """
        return self.local_explainer.visualize(attribution_results, save_path=save_path)
    
    def visualize_global(self, attribution_results, save_path=None):
        """
        Visualize global explanations.
        
        Args:
            attribution_results (dict): Attribution results.
            save_path (str, optional): Path to save the visualization.
            
        Returns:
            matplotlib.figure.Figure: Figure with attribution visualization.
        """
        return self.global_explainer.visualize(attribution_results, save_path=save_path)
    
    def visualize_thermodynamic(self, thermo_results, sequence_idx=0, save_path=None):
        """
        Visualize thermodynamic explanations.
        
        Args:
            thermo_results (dict): Thermodynamic results.
            sequence_idx (int): Index of the sequence to visualize.
            save_path (str, optional): Path to save the visualization.
            
        Returns:
            matplotlib.figure.Figure: Figure with thermodynamic visualization.
        """
        return self.thermodynamic_explainer.visualize(thermo_results, sequence_idx, save_path)
    
    def visualize_counterfactual(self, counterfactual_results, save_path=None):
        """
        Visualize counterfactual explanations.
        
        Args:
            counterfactual_results (dict): Counterfactual results.
            save_path (str, optional): Path to save the visualization.
            
        Returns:
            matplotlib.figure.Figure: Figure with counterfactual visualization.
        """
        return self.counterfactual_explainer.visualize(counterfactual_results, save_path)
