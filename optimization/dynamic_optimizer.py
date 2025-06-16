"""
Dynamic optimization module for the PeptideAI project.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


class FeedbackLoop:
    """
    Real-time feedback loop for model optimization.
    """
    def __init__(self, model, feedback_interval=FEEDBACK_INTERVAL):
        """
        Initialize the feedback loop.
        
        Args:
            model (nn.Module): Model to optimize.
            feedback_interval (int): Interval for feedback collection.
        """
        self.model = model
        self.feedback_interval = feedback_interval
        self.feedback_history = []
        self.prediction_history = []
        self.epoch_history = []
    
    def collect_feedback(self, epoch, sequences, binding_energies):
        """
        Collect feedback on model predictions.
        
        Args:
            epoch (int): Current epoch.
            sequences (list): List of peptide sequences.
            binding_energies (torch.Tensor): True binding energies.
            
        Returns:
            dict: Feedback information.
        """
        # Check if feedback should be collected
        if epoch % self.feedback_interval != 0:
            return None
        
        # Get device from model
        device = next(self.model.parameters()).device
        
        # Move binding_energies to the same device as the model
        binding_energies = binding_energies.to(device)
        
        # Get model predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(sequences)
            predicted_energies = outputs['binding_energy']
        
        # Compute errors
        errors = torch.abs(predicted_energies - binding_energies)
        
        # Identify high-error samples
        high_error_indices = torch.argsort(errors, descending=True)[:5]  # This is correct for PyTorch
        high_error_samples = []
        
        for idx in high_error_indices:
            sample = {
                'sequence': sequences[idx],
                'true_energy': binding_energies[idx].item(),
                'predicted_energy': predicted_energies[idx].item(),
                'error': errors[idx].item(),
                'residue_contributions': outputs['residue_contributions'][idx].cpu().numpy()
            }
            high_error_samples.append(sample)
        
        # Store feedback
        feedback = {
            'epoch': epoch,
            'mean_error': errors.mean().item(),
            'max_error': errors.max().item(),
            'high_error_samples': high_error_samples
        }
        
        self.feedback_history.append(feedback)
        self.prediction_history.append(predicted_energies.cpu().numpy())
        self.epoch_history.append(epoch)
        
        return feedback
    
    def get_optimization_suggestions(self):
        """
        Get suggestions for model optimization based on feedback.
        
        Returns:
            list: List of optimization suggestions.
        """
        if not self.feedback_history:
            return []
        
        suggestions = []
        
        # Check for consistent high errors
        if len(self.feedback_history) >= 3:
            recent_errors = [feedback['mean_error'] for feedback in self.feedback_history[-3:]]
            if all(error > 1.0 for error in recent_errors):
                suggestions.append("Consider increasing model capacity or adjusting learning rate")
        
        # Check for plateauing performance
        if len(self.feedback_history) >= 5:
            recent_errors = [feedback['mean_error'] for feedback in self.feedback_history[-5:]]
            if max(recent_errors) - min(recent_errors) < 0.1:
                suggestions.append("Performance may be plateauing, consider learning rate adjustment")
        
        # Check for specific residue patterns in high-error samples
        if self.feedback_history:
            latest_feedback = self.feedback_history[-1]
            high_error_samples = latest_feedback['high_error_samples']
            
            # Check for common amino acids in high-error samples
            aa_counts = defaultdict(int)
            for sample in high_error_samples:
                for aa in sample['sequence']:
                    aa_counts[aa] += 1
            
            most_common_aa = sorted(aa_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            if most_common_aa:
                aa_str = ', '.join([f"{aa} ({count})" for aa, count in most_common_aa])
                suggestions.append(f"High errors for sequences with amino acids: {aa_str}")
        
        return suggestions
    
    def visualize_feedback(self, save_path=None):
        """
        Visualize feedback history.
        
        Args:
            save_path (str, optional): Path to save the visualization.
            
        Returns:
            matplotlib.figure.Figure: Figure with feedback visualization.
        """
        if not self.feedback_history:
            return None
        
        # Create figure
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot mean error
        epochs = [feedback['epoch'] for feedback in self.feedback_history]
        mean_errors = [feedback['mean_error'] for feedback in self.feedback_history]
        max_errors = [feedback['max_error'] for feedback in self.feedback_history]
        
        axs[0].plot(epochs, mean_errors, 'b-', label='Mean Error')
        axs[0].plot(epochs, max_errors, 'r--', label='Max Error')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Error')
        axs[0].set_title('Feedback Loop: Error Progression')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot prediction distribution
        if self.prediction_history:
            latest_predictions = self.prediction_history[-1]
            axs[1].hist(latest_predictions, bins=20, alpha=0.7)
            axs[1].set_xlabel('Predicted Binding Energy')
            axs[1].set_ylabel('Frequency')
            axs[1].set_title(f'Prediction Distribution (Epoch {epochs[-1]})')
            axs[1].grid(True)
        
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class ContradictionDetector:
    """
    Detector for contradictions in model predictions.
    """
    def __init__(self, model, contradiction_threshold=CONTRADICTION_THRESHOLD):
        """
        Initialize the contradiction detector.
        
        Args:
            model (nn.Module): Model to analyze.
            contradiction_threshold (float): Threshold for contradiction detection.
        """
        self.model = model
        self.contradiction_threshold = contradiction_threshold
        self.contradiction_history = []
    
    def detect_contradictions(self, sequences, binding_energies):
        """
        Detect contradictions in model predictions.
        
        Args:
            sequences (list): List of peptide sequences.
            binding_energies (torch.Tensor): True binding energies.
            
        Returns:
            dict: Contradiction information.
        """
        # Get device from model
        device = next(self.model.parameters()).device
        
        # Move binding_energies to the same device as the model
        binding_energies = binding_energies.to(device)
        
        # Get model predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(sequences)
            predicted_energies = outputs['binding_energy']
            residue_contributions = outputs['residue_contributions']
        
        # Compute sequence similarities
        num_sequences = len(sequences)
        similarities = np.zeros((num_sequences, num_sequences))
        
        for i in range(num_sequences):
            for j in range(i+1, num_sequences):
                # Compute sequence similarity
                seq_i = sequences[i]
                seq_j = sequences[j]
                
                # Use Levenshtein distance for sequence similarity
                similarity = self._compute_sequence_similarity(seq_i, seq_j)
                similarities[i, j] = similarity
                similarities[j, i] = similarity
        
        # Find contradictions
        contradictions = []
        
        for i in range(num_sequences):
            for j in range(i+1, num_sequences):
                # Check if sequences are similar
                if similarities[i, j] > self.contradiction_threshold:
                    # Check if binding energies are significantly different
                    energy_diff = abs(binding_energies[i].item() - binding_energies[j].item())
                    pred_diff = abs(predicted_energies[i].item() - predicted_energies[j].item())
                    
                    # Check for contradiction
                    if (energy_diff < 0.5 and pred_diff > 2.0) or (energy_diff > 2.0 and pred_diff < 0.5):
                        contradiction = {
                            'sequence_i': sequences[i],
                            'sequence_j': sequences[j],
                            'similarity': similarities[i, j],
                            'true_energy_i': binding_energies[i].item(),
                            'true_energy_j': binding_energies[j].item(),
                            'pred_energy_i': predicted_energies[i].item(),
                            'pred_energy_j': predicted_energies[j].item(),
                            'energy_diff': energy_diff,
                            'pred_diff': pred_diff,
                            'residue_contrib_i': residue_contributions[i].cpu().numpy(),
                            'residue_contrib_j': residue_contributions[j].cpu().numpy()
                        }
                        contradictions.append(contradiction)
        
        # Store contradictions
        contradiction_info = {
            'num_contradictions': len(contradictions),
            'contradictions': contradictions
        }
        
        self.contradiction_history.append(contradiction_info)
        
        return contradiction_info
    
    def _compute_sequence_similarity(self, seq_i, seq_j):
        """
        Compute similarity between two sequences.
        
        Args:
            seq_i (str): First sequence.
            seq_j (str): Second sequence.
            
        Returns:
            float: Similarity score.
        """
        # Compute Levenshtein distance
        distance = self._levenshtein_distance(seq_i, seq_j)
        
        # Normalize by maximum length
        max_length = max(len(seq_i), len(seq_j))
        similarity = 1.0 - distance / max_length
        
        return similarity
    
    def _levenshtein_distance(self, s1, s2):
        """
        Compute Levenshtein distance between two strings.
        
        Args:
            s1 (str): First string.
            s2 (str): Second string.
            
        Returns:
            int: Levenshtein distance.
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def get_resolution_strategies(self):
        """
        Get strategies to resolve contradictions.
        
        Returns:
            list: List of resolution strategies.
        """
        if not self.contradiction_history or self.contradiction_history[-1]['num_contradictions'] == 0:
            return []
        
        strategies = []
        latest_contradictions = self.contradiction_history[-1]['contradictions']
        
        # Analyze contradictions
        for i, contradiction in enumerate(latest_contradictions):
            seq_i = contradiction['sequence_i']
            seq_j = contradiction['sequence_j']
            
            # Find differing positions
            diff_positions = []
            for pos, (aa_i, aa_j) in enumerate(zip(seq_i, seq_j)):
                if aa_i != aa_j:
                    diff_positions.append((pos, aa_i, aa_j))
            
            # Generate strategy
            if diff_positions:
                diff_str = ', '.join([f"position {pos}: {aa_i} vs {aa_j}" for pos, aa_i, aa_j in diff_positions])
                strategy = f"Contradiction {i+1}: Sequences differ at {diff_str}. "
                
                # Check residue contributions
                contrib_i = contradiction['residue_contrib_i']
                contrib_j = contradiction['residue_contrib_j']
                
                # Find positions with high contribution differences
                high_diff_positions = []
                for pos, (c_i, c_j) in enumerate(zip(contrib_i[:len(seq_i)], contrib_j[:len(seq_j)])):
                    if abs(c_i - c_j) > 0.5:
                        high_diff_positions.append((pos, c_i, c_j))
                
                if high_diff_positions:
                    high_diff_str = ', '.join([f"position {pos}: {c_i:.2f} vs {c_j:.2f}" for pos, c_i, c_j in high_diff_positions])
                    strategy += f"High contribution differences at {high_diff_str}. "
                
                # Suggest resolution
                strategy += "Consider adding more similar sequences to the training data."
                strategies.append(strategy)
        
        return strategies
    
    def visualize_contradictions(self, save_path=None):
        """
        Visualize contradictions.
        
        Args:
            save_path (str, optional): Path to save the visualization.
            
        Returns:
            matplotlib.figure.Figure: Figure with contradiction visualization.
        """
        if not self.contradiction_history or self.contradiction_history[-1]['num_contradictions'] == 0:
            return None
        
        # Get latest contradictions
        latest_contradictions = self.contradiction_history[-1]['contradictions']
        num_contradictions = min(len(latest_contradictions), 3)  # Show at most 3 contradictions
        
        # Create figure
        fig, axs = plt.subplots(num_contradictions, 2, figsize=(12, 4 * num_contradictions))
        
        # Handle case with only one contradiction
        if num_contradictions == 1:
            axs = [axs]
        
        # Plot contradictions
        for i in range(num_contradictions):
            contradiction = latest_contradictions[i]
            
            # Get data
            seq_i = contradiction['sequence_i']
            seq_j = contradiction['sequence_j']
            contrib_i = contradiction['residue_contrib_i'][:len(seq_i)]
            contrib_j = contradiction['residue_contrib_j'][:len(seq_j)]
            
            # Plot residue contributions for sequence i
            axs[i][0].bar(range(len(seq_i)), contrib_i)
            axs[i][0].set_xticks(range(len(seq_i)))
            axs[i][0].set_xticklabels(list(seq_i))
            axs[i][0].set_xlabel('Residue')
            axs[i][0].set_ylabel('Contribution')
            axs[i][0].set_title(f"Sequence {i+1}A: {seq_i}\nTrue: {contradiction['true_energy_i']:.2f}, Pred: {contradiction['pred_energy_i']:.2f}")
            axs[i][0].grid(True)
            
            # Plot residue contributions for sequence j
            axs[i][1].bar(range(len(seq_j)), contrib_j)
            axs[i][1].set_xticks(range(len(seq_j)))
            axs[i][1].set_xticklabels(list(seq_j))
            axs[i][1].set_xlabel('Residue')
            axs[i][1].set_ylabel('Contribution')
            axs[i][1].set_title(f"Sequence {i+1}B: {seq_j}\nTrue: {contradiction['true_energy_j']:.2f}, Pred: {contradiction['pred_energy_j']:.2f}")
            axs[i][1].grid(True)
        
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class DynamicOptimizer:
    """
    Dynamic optimizer for model optimization.
    """
    def __init__(self, model, feedback_interval=FEEDBACK_INTERVAL, contradiction_threshold=CONTRADICTION_THRESHOLD):
        """
        Initialize the dynamic optimizer.
        
        Args:
            model (nn.Module): Model to optimize.
            feedback_interval (int): Interval for feedback collection.
            contradiction_threshold (float): Threshold for contradiction detection.
        """
        self.model = model
        self.feedback_loop = FeedbackLoop(model, feedback_interval)
        self.contradiction_detector = ContradictionDetector(model, contradiction_threshold)
    
    def step(self, epoch, sequences, binding_energies):
        """
        Perform optimization step.
        
        Args:
            epoch (int): Current epoch.
            sequences (list): List of peptide sequences.
            binding_energies (torch.Tensor): True binding energies.
            
        Returns:
            dict: Optimization information.
        """
        # Get device from model
        device = next(self.model.parameters()).device
        
        # Move binding_energies to the same device as the model
        binding_energies = binding_energies.to(device)
        
        # Collect feedback
        feedback = self.feedback_loop.collect_feedback(epoch, sequences, binding_energies)
        
        # Detect contradictions
        contradiction_info = self.contradiction_detector.detect_contradictions(sequences, binding_energies)
        
        # Get optimization suggestions
        suggestions = self.feedback_loop.get_optimization_suggestions()
        
        # Get resolution strategies
        strategies = self.contradiction_detector.get_resolution_strategies()
        
        # Combine information
        optimization_info = {
            'epoch': epoch,
            'feedback': feedback,
            'contradictions': contradiction_info,
            'suggestions': suggestions,
            'strategies': strategies
        }
        
        return optimization_info
    
    def visualize(self, save_dir=None):
        """
        Visualize optimization information.
        
        Args:
            save_dir (str, optional): Directory to save visualizations.
            
        Returns:
            tuple: Tuple of figures.
        """
        # Create save directory if provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Visualize feedback
        feedback_path = os.path.join(save_dir, 'feedback.png') if save_dir else None
        feedback_fig = self.feedback_loop.visualize_feedback(feedback_path)
        
        # Visualize contradictions
        contradiction_path = os.path.join(save_dir, 'contradictions.png') if save_dir else None
        contradiction_fig = self.contradiction_detector.visualize_contradictions(contradiction_path)
        
        return feedback_fig, contradiction_fig
