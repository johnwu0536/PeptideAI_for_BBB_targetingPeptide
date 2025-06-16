"""
Visualization utilities for the PeptideAI project.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import torch
import dgl

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


class PeptideVisualizer:
    """
    Visualizer for peptide sequences and binding energies.
    """
    def __init__(self, output_dir=VISUALIZATION_DIR):
        """
        Initialize the visualizer.
        
        Args:
            output_dir (str): Directory to save visualizations.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up color maps
        self.setup_colormaps()
    
    def setup_colormaps(self):
        """
        Set up color maps for visualization.
        """
        # Color map for amino acid properties
        self.aa_property_colors = {
            'hydrophobic': '#1E88E5',  # Blue
            'polar': '#FFC107',        # Yellow
            'positive': '#D81B60',     # Red
            'negative': '#004D40',     # Green
            'special': '#8E24AA'       # Purple
        }
        
        # Classify amino acids by property
        self.aa_properties = {
            'A': 'hydrophobic',  # Alanine
            'C': 'polar',        # Cysteine
            'D': 'negative',     # Aspartic acid
            'E': 'negative',     # Glutamic acid
            'F': 'hydrophobic',  # Phenylalanine
            'G': 'special',      # Glycine
            'H': 'positive',     # Histidine
            'I': 'hydrophobic',  # Isoleucine
            'K': 'positive',     # Lysine
            'L': 'hydrophobic',  # Leucine
            'M': 'hydrophobic',  # Methionine
            'N': 'polar',        # Asparagine
            'P': 'special',      # Proline
            'Q': 'polar',        # Glutamine
            'R': 'positive',     # Arginine
            'S': 'polar',        # Serine
            'T': 'polar',        # Threonine
            'V': 'hydrophobic',  # Valine
            'W': 'hydrophobic',  # Tryptophan
            'Y': 'polar'         # Tyrosine
        }
        
        # Color map for residue contributions
        self.contribution_cmap = plt.cm.coolwarm
    
    def visualize_sequence(self, sequence, predicted_energy, residue_contributions, title=None, save_path=None):
        """
        Visualize a peptide sequence with residue contributions.
        
        Args:
            sequence (str): Peptide sequence.
            predicted_energy (float): Predicted binding energy.
            residue_contributions (numpy.ndarray): Residue contributions.
            title (str, optional): Title for the plot.
            save_path (str, optional): Path to save the visualization.
            
        Returns:
            matplotlib.figure.Figure: Figure with sequence visualization.
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot residue contributions
        bars = ax.bar(range(len(sequence)), residue_contributions)
        
        # Color bars by contribution
        for i, bar in enumerate(bars):
            bar.set_color(self.contribution_cmap(0.5 * (1 + residue_contributions[i] / max(abs(residue_contributions)))))
        
        # Add amino acid labels
        ax.set_xticks(range(len(sequence)))
        ax.set_xticklabels(list(sequence))
        
        # Add property colors to x-tick labels
        for i, aa in enumerate(sequence):
            ax.get_xticklabels()[i].set_color(self.aa_property_colors[self.aa_properties[aa]])
        
        # Add labels and title
        ax.set_xlabel('Residue')
        ax.set_ylabel('Contribution')
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Sequence: {sequence}, Predicted Energy: {predicted_energy:.2f}")
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=self.contribution_cmap, norm=plt.Normalize(vmin=-max(abs(residue_contributions)), vmax=max(abs(residue_contributions))))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Contribution Value')
        
        # Add legend for amino acid properties
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=prop.capitalize())
                          for prop, color in self.aa_property_colors.items()]
        ax.legend(handles=legend_elements, title="AA Properties", loc='upper right')
        
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_binding_energy_distribution(self, binding_energies, title=None, save_path=None):
        """
        Visualize the distribution of binding energies.
        
        Args:
            binding_energies (numpy.ndarray): Binding energies.
            title (str, optional): Title for the plot.
            save_path (str, optional): Path to save the visualization.
            
        Returns:
            matplotlib.figure.Figure: Figure with binding energy distribution.
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram
        sns.histplot(binding_energies, bins=20, kde=True, ax=ax)
        
        # Add labels and title
        ax.set_xlabel('Binding Energy (kcal/mol)')
        ax.set_ylabel('Frequency')
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Binding Energy Distribution')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add statistics
        mean = np.mean(binding_energies)
        std = np.std(binding_energies)
        median = np.median(binding_energies)
        
        stats_text = f"Mean: {mean:.2f}\nStd: {std:.2f}\nMedian: {median:.2f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_peptide_graph(self, graph, sequence, residue_contributions=None, save_path=None):
        """
        Visualize the graph representation of a peptide.
        
        Args:
            graph (dgl.DGLGraph): Graph representation of the peptide.
            sequence (str): Peptide sequence.
            residue_contributions (numpy.ndarray, optional): Residue contributions.
            save_path (str, optional): Path to save the visualization.
            
        Returns:
            matplotlib.figure.Figure: Figure with peptide graph visualization.
        """
        # Convert graph to NetworkX graph
        if isinstance(graph, dgl.DGLGraph):
            nx_graph = graph.to_networkx().to_undirected()
        elif hasattr(graph, 'to_networkx'):
            # PyTorch Geometric graph
            nx_graph = graph.to_networkx().to_undirected()
        elif hasattr(graph, 'edge_index'):
            # PyTorch Geometric Data object
            import torch_geometric.utils as pyg_utils
            nx_graph = pyg_utils.to_networkx(graph, to_undirected=True)
        else:
            # Assume it's already a NetworkX graph
            nx_graph = graph
            
        # Ensure nx_graph is a NetworkX graph
        if not isinstance(nx_graph, nx.Graph):
            print(f"Warning: Could not convert graph of type {type(graph)} to NetworkX graph")
            # Create a simple chain graph as fallback
            nx_graph = nx.path_graph(len(sequence))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get node positions (circular layout)
        pos = nx.circular_layout(nx_graph)
        
        # Get the number of nodes in the graph
        num_nodes = len(nx_graph.nodes())
        
        # Ensure sequence length matches the number of nodes
        if len(sequence) != num_nodes:
            print(f"Warning: Sequence length ({len(sequence)}) doesn't match number of nodes ({num_nodes})")
            # Adjust sequence to match number of nodes
            if len(sequence) > num_nodes:
                sequence = sequence[:num_nodes]
            else:
                # Pad sequence with 'X' if needed
                sequence = sequence + 'X' * (num_nodes - len(sequence))
        
        # Set node colors based on amino acid properties
        node_colors = []
        for i, aa in enumerate(sequence):
            if aa in self.aa_properties:
                node_colors.append(self.aa_property_colors[self.aa_properties[aa]])
            else:
                # Default color for unknown amino acids
                node_colors.append('gray')
        
        # Set node sizes based on residue contributions if provided
        if residue_contributions is not None:
            # Ensure residue_contributions length matches number of nodes
            if len(residue_contributions) != num_nodes:
                print(f"Warning: Residue contributions length ({len(residue_contributions)}) doesn't match number of nodes ({num_nodes})")
                # Adjust residue_contributions to match number of nodes
                if len(residue_contributions) > num_nodes:
                    residue_contributions = residue_contributions[:num_nodes]
                else:
                    # Pad residue_contributions with zeros if needed
                    residue_contributions = np.concatenate([residue_contributions, np.zeros(num_nodes - len(residue_contributions))])
            
            # Normalize contributions to range [50, 500]
            max_contrib = max(abs(residue_contributions))
            if max_contrib > 0:  # Avoid division by zero
                node_sizes = [50 + 450 * abs(contrib) / max_contrib for contrib in residue_contributions]
            else:
                node_sizes = [300] * num_nodes
        else:
            node_sizes = [300] * num_nodes
        
        # Ensure node_colors and node_sizes have the same length as the number of nodes
        if len(node_colors) != num_nodes:
            node_colors = node_colors[:num_nodes] if len(node_colors) > num_nodes else node_colors + ['gray'] * (num_nodes - len(node_colors))
        
        if len(node_sizes) != num_nodes:
            node_sizes = node_sizes[:num_nodes] if len(node_sizes) > num_nodes else node_sizes + [300] * (num_nodes - len(node_sizes))
        
        # Draw nodes
        nx.draw_networkx_nodes(nx_graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, ax=ax)
        
        # Draw edges
        edge_types = []
        for u, v in nx_graph.edges():
            if abs(u - v) == 1:
                edge_types.append('backbone')
            else:
                edge_types.append('spatial')
        
        backbone_edges = [e for i, e in enumerate(nx_graph.edges()) if edge_types[i] == 'backbone']
        spatial_edges = [e for i, e in enumerate(nx_graph.edges()) if edge_types[i] == 'spatial']
        
        nx.draw_networkx_edges(nx_graph, pos, edgelist=backbone_edges, width=2, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(nx_graph, pos, edgelist=spatial_edges, width=1, alpha=0.5, style='dashed', ax=ax)
        
        # Draw labels
        labels = {i: f"{sequence[i]}{i+1}" for i in range(len(sequence))}
        nx.draw_networkx_labels(nx_graph, pos, labels=labels, font_size=10, font_weight='bold', ax=ax)
        
        # Add title
        ax.set_title(f"Graph Representation of Peptide: {sequence}")
        
        # Remove axis
        ax.axis('off')
        
        # Add legend for amino acid properties
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=prop.capitalize())
                          for prop, color in self.aa_property_colors.items()]
        
        # Add legend for edge types
        from matplotlib.lines import Line2D
        legend_elements.extend([
            Line2D([0], [0], color='black', lw=2, label='Backbone'),
            Line2D([0], [0], color='black', lw=1, linestyle='dashed', label='Spatial')
        ])
        
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_attention_weights(self, attention_weights, sequence, save_path=None):
        """
        Visualize attention weights.
        
        Args:
            attention_weights (torch.Tensor): Attention weights.
            sequence (str): Peptide sequence.
            save_path (str, optional): Path to save the visualization.
            
        Returns:
            matplotlib.figure.Figure: Figure with attention weights visualization.
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Convert attention weights to numpy array
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.cpu().numpy()
        
        # Plot heatmap
        sns.heatmap(attention_weights, annot=True, cmap='viridis', ax=ax)
        
        # Add labels
        ax.set_xlabel('Target Position')
        ax.set_ylabel('Source Position')
        
        # Add sequence labels
        ax.set_xticks(np.arange(len(sequence)) + 0.5)
        ax.set_yticks(np.arange(len(sequence)) + 0.5)
        ax.set_xticklabels(list(sequence))
        ax.set_yticklabels(list(sequence))
        
        # Add title
        ax.set_title('Attention Weights')
        
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_optimization_progress(self, epochs, metrics, metric_name, save_path=None):
        """
        Visualize optimization progress.
        
        Args:
            epochs (list): List of epochs.
            metrics (dict): Dictionary of metrics.
            metric_name (str): Name of the metric to visualize.
            save_path (str, optional): Path to save the visualization.
            
        Returns:
            matplotlib.figure.Figure: Figure with optimization progress visualization.
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot metrics
        for name, values in metrics.items():
            ax.plot(epochs, values, label=name)
        
        # Add labels and title
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name.capitalize())
        ax.set_title(f'{metric_name.capitalize()} vs. Epoch')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        ax.legend()
        
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_3d_embedding(self, embeddings, sequences, binding_energies, save_path=None):
        """
        Visualize peptide embeddings in 3D space.
        
        Args:
            embeddings (numpy.ndarray): Peptide embeddings.
            sequences (list): List of peptide sequences.
            binding_energies (numpy.ndarray): Binding energies.
            save_path (str, optional): Path to save the visualization.
            
        Returns:
            matplotlib.figure.Figure: Figure with 3D embedding visualization.
        """
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Normalize binding energies to [0, 1] for coloring
        norm_energies = (binding_energies - binding_energies.min()) / (binding_energies.max() - binding_energies.min())
        
        # Plot embeddings
        scatter = ax.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            embeddings[:, 2],
            c=binding_energies,
            cmap='coolwarm',
            s=100,
            alpha=0.8
        )
        
        # Add labels
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        
        # Add title
        ax.set_title('3D Visualization of Peptide Embeddings')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Binding Energy (kcal/mol)')
        
        # Add annotations for a few points
        for i in range(min(5, len(sequences))):
            ax.text(
                embeddings[i, 0],
                embeddings[i, 1],
                embeddings[i, 2],
                sequences[i],
                size=8,
                zorder=1,
                color='k'
            )
        
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_sequence_comparison(self, sequences, binding_energies, residue_contributions, save_path=None):
        """
        Visualize comparison between multiple peptide sequences.
        
        Args:
            sequences (list): List of peptide sequences.
            binding_energies (list): List of binding energies.
            residue_contributions (list): List of residue contributions.
            save_path (str, optional): Path to save the visualization.
            
        Returns:
            matplotlib.figure.Figure: Figure with sequence comparison visualization.
        """
        # Limit to at most 5 sequences
        num_sequences = min(5, len(sequences))
        sequences = sequences[:num_sequences]
        binding_energies = binding_energies[:num_sequences]
        residue_contributions = residue_contributions[:num_sequences]
        
        # Create figure
        fig, axs = plt.subplots(num_sequences, 1, figsize=(12, 4 * num_sequences))
        
        # Handle case with only one sequence
        if num_sequences == 1:
            axs = [axs]
        
        # Plot each sequence
        for i, (seq, energy, contrib) in enumerate(zip(sequences, binding_energies, residue_contributions)):
            # Plot residue contributions
            bars = axs[i].bar(range(len(seq)), contrib)
            
            # Color bars by contribution
            for j, bar in enumerate(bars):
                bar.set_color(self.contribution_cmap(0.5 * (1 + contrib[j] / max(abs(contrib)))))
            
            # Add amino acid labels
            axs[i].set_xticks(range(len(seq)))
            axs[i].set_xticklabels(list(seq))
            
            # Add property colors to x-tick labels
            for j, aa in enumerate(seq):
                axs[i].get_xticklabels()[j].set_color(self.aa_property_colors[self.aa_properties[aa]])
            
            # Add labels and title
            axs[i].set_xlabel('Residue')
            axs[i].set_ylabel('Contribution')
            axs[i].set_title(f"Sequence {i+1}: {seq}, Binding Energy: {energy:.2f}")
            
            # Add grid
            axs[i].grid(True, linestyle='--', alpha=0.7)
        
        # Add legend for amino acid properties
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=prop.capitalize())
                          for prop, color in self.aa_property_colors.items()]
        axs[0].legend(handles=legend_elements, title="AA Properties", loc='upper right')
        
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_sequence_logo(self, sequences, weights=None, save_path=None):
        """
        Visualize a sequence logo for a set of peptide sequences.
        
        Args:
            sequences (list): List of peptide sequences.
            weights (list, optional): List of weights for each sequence.
            save_path (str, optional): Path to save the visualization.
            
        Returns:
            matplotlib.figure.Figure: Figure with sequence logo visualization.
        """
        # Ensure all sequences have the same length
        max_length = max(len(seq) for seq in sequences)
        padded_sequences = [seq.ljust(max_length, '-') for seq in sequences]
        
        # Count amino acid frequencies at each position
        aa_counts = np.zeros((max_length, len(AMINO_ACIDS) + 1))  # +1 for gap
        
        for i, seq in enumerate(padded_sequences):
            for j, aa in enumerate(seq):
                if aa == '-':
                    aa_index = len(AMINO_ACIDS)
                else:
                    aa_index = AMINO_ACIDS.index(aa)
                
                if weights is not None:
                    aa_counts[j, aa_index] += weights[i]
                else:
                    aa_counts[j, aa_index] += 1
        
        # Normalize counts to get probabilities
        aa_probs = aa_counts / aa_counts.sum(axis=1, keepdims=True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(max_length * 0.5 + 2, 6))
        
        # Plot sequence logo
        for i in range(max_length):
            y_offset = 0
            for j, aa in enumerate(AMINO_ACIDS + '-'):
                if aa == '-':
                    continue
                
                height = aa_probs[i, j if j < len(AMINO_ACIDS) else -1]
                if height > 0:
                    if aa in self.aa_properties:
                        color = self.aa_property_colors[self.aa_properties[aa]]
                    else:
                        color = 'gray'
                    
                    ax.text(i, y_offset + height/2, aa, ha='center', va='center',
                           fontsize=12 + 24 * height, fontweight='bold', color='white',
                           bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
                    
                    y_offset += height
        
        # Set axis limits
        ax.set_xlim(-0.5, max_length - 0.5)
        ax.set_ylim(0, 1)
        
        # Add labels and title
        ax.set_xlabel('Position')
        ax.set_ylabel('Probability')
        ax.set_title('Sequence Logo')
        
        # Add x-ticks
        ax.set_xticks(range(max_length))
        ax.set_xticklabels([str(i+1) for i in range(max_length)])
        
        # Remove y-ticks
        ax.set_yticks([])
        
        # Add legend for amino acid properties
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=prop.capitalize())
                          for prop, color in self.aa_property_colors.items()]
        ax.legend(handles=legend_elements, title="AA Properties", loc='upper right')
        
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
