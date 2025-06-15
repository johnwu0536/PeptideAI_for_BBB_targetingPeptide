"""
Main script for the PeptideAI project.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import json

from config import *
from utils.data_processing import (
    load_data, prepare_datasets, get_advanced_dataloaders, create_batch_graphs
)
from models.model import PeptideBindingModel
from explainability.explainers import ExplainabilityManager
from optimization.dynamic_optimizer import DynamicOptimizer
from visualization.visualizer import PeptideVisualizer


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='PeptideAI: Explainable AI for Peptide Binding Energy Prediction')
    
    # General settings
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'explain'],
                        help='Mode to run the script in')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help='Random seed for reproducibility')
    
    # Data settings
    parser.add_argument('--data_path', type=str, default=DATA_PATH,
                        help='Path to the data file')
    
    # Model settings
    parser.add_argument('--embedding_dim', type=int, default=SEQ_EMBEDDING_DIM,
                        help='Dimension of the embeddings')
    parser.add_argument('--num_heads', type=int, default=SEQ_NUM_HEADS,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=SEQ_NUM_LAYERS,
                        help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=SEQ_DROPOUT,
                        help='Dropout rate')
    
    # Training settings
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY,
                        help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS,
                        help='Number of epochs')
    parser.add_argument('--early_stopping', type=int, default=EARLY_STOPPING_PATIENCE,
                        help='Patience for early stopping')
    
    # Optimization settings
    parser.add_argument('--feedback_interval', type=int, default=FEEDBACK_INTERVAL,
                        help='Interval for feedback collection')
    parser.add_argument('--contradiction_threshold', type=float, default=CONTRADICTION_THRESHOLD,
                        help='Threshold for contradiction detection')
    
    # Explainability settings
    parser.add_argument('--explain_method', type=str, default='integrated_gradients',
                        choices=['integrated_gradients', 'deep_lift', 'gradient_shap', 'occlusion'],
                        help='Explainability method')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='Directory to save outputs')
    
    # Model path - explicitly separated from other arguments to avoid parsing conflicts
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to load a pre-trained model')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Print parsed arguments for debugging
    print("Parsed arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
    return args


def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args):
    """
    Train the model.
    
    Args:
        args (argparse.Namespace): Command line arguments.
    """
    # Set device
    device = torch.device(args.device)
    
    # Load data
    print("Loading data...")
    df = load_data(args.data_path)
    
    # Prepare datasets
    print("Preparing datasets...")
    train_dataset, val_dataset, test_dataset = prepare_datasets(df)
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = get_advanced_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size=args.batch_size
    )
    
    # Initialize model
    print("Initializing model...")
    model = PeptideBindingModel(
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Initialize learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Initialize loss function
    criterion = nn.MSELoss()
    
    # Initialize dynamic optimizer
    dynamic_optimizer = DynamicOptimizer(
        model,
        feedback_interval=args.feedback_interval,
        contradiction_threshold=args.contradiction_threshold
    )
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=LOG_DIR)
    
    # Initialize visualizer
    visualizer = PeptideVisualizer(output_dir=VISUALIZATION_DIR)
    
    # Initialize explainability manager
    explainability_manager = ExplainabilityManager(model)
    
    # Initialize training metrics
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    epochs = []
    
    # Training loop
    print("Starting training...")
    for epoch in range(args.num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]")
        
        for batch in train_pbar:
            # Get batch data
            sequences = batch['sequences']
            binding_energies = batch['binding_energies'].to(device)
            graph = batch['graph'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(sequences, graph)
            predicted_energies = outputs['binding_energy']
            
            # Compute loss
            loss = criterion(predicted_energies, binding_energies)
            
            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VAL)
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item() * len(sequences)
            
            # Update progress bar
            train_pbar.set_postfix({'loss': loss.item()})
        
        # Compute average training loss
        train_loss /= len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        # Progress bar for validation
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Val]")
        
        with torch.no_grad():
            for batch in val_pbar:
                # Get batch data
                sequences = batch['sequences']
                binding_energies = batch['binding_energies'].to(device)
                graph = batch['graph'].to(device)
                
                # Forward pass
                outputs = model(sequences, graph)
                predicted_energies = outputs['binding_energy']
                
                # Compute loss
                loss = criterion(predicted_energies, binding_energies)
                
                # Update metrics
                val_loss += loss.item() * len(sequences)
                
                # Update progress bar
                val_pbar.set_postfix({'loss': loss.item()})
        
        # Compute average validation loss
        val_loss /= len(val_loader.dataset)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Update dynamic optimizer
        dynamic_opt_info = dynamic_optimizer.step(
            epoch,
            train_dataset.sequences[:args.batch_size],
            torch.tensor(train_dataset.binding_energies[:args.batch_size], device=device)
        )
        
        # Log metrics
        print(f"Epoch {epoch+1}/{args.num_epochs}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Save metrics for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epochs.append(epoch)
        
        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': vars(args)
            }, os.path.join(MODEL_DIR, 'best_model.pth'))
            
            print(f"Saved best model with validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Visualize training progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            # Plot loss curves
            fig = visualizer.visualize_optimization_progress(
                epochs, {'loss': train_losses, 'val_loss': val_losses}, 'loss',
                save_path=os.path.join(VISUALIZATION_DIR, f'loss_epoch_{epoch+1}.png')
            )
            plt.close(fig)
            
            # Visualize sample predictions
            sample_batch = next(iter(val_loader))
            sample_sequences = sample_batch['sequences'][:5]
            sample_binding_energies = sample_batch['binding_energies'][:5].cpu().numpy()
            sample_graph = sample_batch['graph'].to(device)
            
            with torch.no_grad():
                sample_outputs = model(sample_sequences, sample_graph)
                sample_predictions = sample_outputs['binding_energy'].cpu().numpy()
                sample_residue_contributions = sample_outputs['residue_contributions'].cpu().numpy()
            
            for i, (seq, true, pred, contrib) in enumerate(zip(
                sample_sequences, sample_binding_energies, sample_predictions, sample_residue_contributions
            )):
                fig = visualizer.visualize_sequence(
                    seq, pred, contrib,
                    title=f"True: {true:.2f}, Pred: {pred:.2f}",
                    save_path=os.path.join(VISUALIZATION_DIR, f'sample_{i}_epoch_{epoch+1}.png')
                )
                plt.close(fig)
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'args': vars(args)
    }, os.path.join(MODEL_DIR, 'final_model.pth'))
    
    print(f"Saved final model with validation loss: {val_loss:.4f}")
    
    # Test the model
    test(args, model, test_loader, device, visualizer, explainability_manager)
    
    # Close tensorboard writer
    writer.close()


def test(args, model=None, test_loader=None, device=None, visualizer=None, explainability_manager=None):
    """
    Test the model.
    
    Args:
        args (argparse.Namespace): Command line arguments.
        model (nn.Module, optional): Model to test. If None, load from args.model_path.
        test_loader (DataLoader, optional): Test dataloader. If None, create from args.data_path.
        device (torch.device, optional): Device to use. If None, use args.device.
        visualizer (PeptideVisualizer, optional): Visualizer. If None, create a new one.
        explainability_manager (ExplainabilityManager, optional): Explainability manager. If None, create a new one.
    """
    # Set device if not provided
    if device is None:
        device = torch.device(args.device)
    
    # Load model if not provided
    if model is None:
        print("Loading model...")
        model_path = args.model_path or os.path.join(MODEL_DIR, 'best_model.pth')
        checkpoint = torch.load(model_path, map_location=device)
        
        # Initialize model
        model = PeptideBindingModel(
            embedding_dim=args.embedding_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout
        ).to(device)
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Loaded model from {model_path}")
    
    # Load data if test_loader not provided
    if test_loader is None:
        print("Loading data...")
        df = load_data(args.data_path)
        
        # Prepare datasets
        print("Preparing datasets...")
        _, _, test_dataset = prepare_datasets(df)
        
        # Create dataloaders
        print("Creating dataloaders...")
        _, _, test_loader = get_advanced_dataloaders(
            None, None, test_dataset, batch_size=args.batch_size
        )
    
    # Initialize visualizer if not provided
    if visualizer is None:
        visualizer = PeptideVisualizer(output_dir=VISUALIZATION_DIR)
    
    # Initialize explainability manager if not provided
    if explainability_manager is None:
        explainability_manager = ExplainabilityManager(model)
    
    # Initialize loss function
    criterion = nn.MSELoss()
    
    # Test the model
    print("Testing model...")
    model.eval()
    test_loss = 0.0
    all_predictions = []
    all_true_values = []
    all_sequences = []
    
    # Progress bar for testing
    test_pbar = tqdm(test_loader, desc="Testing")
    
    with torch.no_grad():
        for batch in test_pbar:
            # Get batch data
            sequences = batch['sequences']
            binding_energies = batch['binding_energies'].to(device)
            graph = batch['graph'].to(device)
            
            # Forward pass
            outputs = model(sequences, graph)
            predicted_energies = outputs['binding_energy']
            
            # Compute loss
            loss = criterion(predicted_energies, binding_energies)
            
            # Update metrics
            test_loss += loss.item() * len(sequences)
            
            # Store predictions and true values
            all_predictions.extend(predicted_energies.cpu().numpy())
            all_true_values.extend(binding_energies.cpu().numpy())
            all_sequences.extend(sequences)
            
            # Update progress bar
            test_pbar.set_postfix({'loss': loss.item()})
    
    # Compute average test loss
    test_loss /= len(test_loader.dataset)
    
    # Compute metrics
    mae = np.mean(np.abs(np.array(all_predictions) - np.array(all_true_values)))
    rmse = np.sqrt(np.mean(np.square(np.array(all_predictions) - np.array(all_true_values))))
    r2 = 1 - np.sum(np.square(np.array(all_predictions) - np.array(all_true_values))) / np.sum(np.square(np.array(all_true_values) - np.mean(all_true_values)))
    
    # Print metrics
    print(f"Test Loss: {test_loss:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Save metrics - convert to native Python types for JSON serialization
    metrics = {
        'test_loss': float(test_loss),
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2)
    }
    
    # Add additional metrics for debugging
    metrics['predictions_sample'] = [float(p) for p in all_predictions[:5]]
    metrics['true_values_sample'] = [float(t) for t in all_true_values[:5]]
    
    try:
        with open(os.path.join(RESULTS_DIR, 'test_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        print("Successfully saved metrics to JSON file")
    except TypeError as e:
        print(f"Error saving metrics to JSON: {e}")
        print("Attempting to convert all values to native Python types...")
        
        # Convert all values to native Python types
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (np.ndarray, np.number)):
                serializable_metrics[key] = value.item() if np.isscalar(value) else [float(v) for v in value]
            elif isinstance(value, torch.Tensor):
                serializable_metrics[key] = value.item() if value.numel() == 1 else [float(v) for v in value.cpu().numpy()]
            else:
                serializable_metrics[key] = value
        
        with open(os.path.join(RESULTS_DIR, 'test_metrics.json'), 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
        print("Successfully saved converted metrics to JSON file")
    
    # Visualize test results
    # Plot true vs predicted
    plt.figure(figsize=(10, 8))
    plt.scatter(all_true_values, all_predictions, alpha=0.7)
    plt.plot([min(all_true_values), max(all_true_values)], [min(all_true_values), max(all_true_values)], 'k--')
    plt.xlabel('True Binding Energy (kcal/mol)')
    plt.ylabel('Predicted Binding Energy (kcal/mol)')
    plt.title(f'True vs Predicted Binding Energy (R² = {r2:.4f})')
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'true_vs_predicted.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot binding energy distribution
    fig = visualizer.visualize_binding_energy_distribution(
        np.array(all_predictions),
        title="Predicted Binding Energy Distribution",
        save_path=os.path.join(VISUALIZATION_DIR, 'binding_energy_distribution.png')
    )
    plt.close(fig)
    
    # Visualize sample sequences with explanations
    num_samples = min(5, len(all_sequences))
    sample_indices = np.random.choice(len(all_sequences), num_samples, replace=False)
    
    for i, idx in enumerate(sample_indices):
        sequence = all_sequences[idx]
        true_energy = all_true_values[idx]
        predicted_energy = all_predictions[idx]
        
        # Create graph for the sequence
        graph = create_batch_graphs([sequence])[0].to(device)
        
        # Get local explanation
        local_explanation = explainability_manager.explain_local(
            [sequence], [graph], method=args.explain_method
        )
        
        # Get thermodynamic map
        thermo_map = explainability_manager.explain_thermodynamic(
            [sequence], [graph]
        )
        
        # Get counterfactual explanation
        counterfactual_explanation = explainability_manager.explain_counterfactual(
            sequence, graph
        )
        
        # Visualize sequence with explanation
        residue_contributions = local_explanation['residue_attributions'][0].cpu().numpy()
        
        fig = visualizer.visualize_sequence(
            sequence, predicted_energy, residue_contributions,
            title=f"True: {true_energy:.2f}, Pred: {predicted_energy:.2f}",
            save_path=os.path.join(VISUALIZATION_DIR, f'explanation_sequence_{i}.png')
        )
        plt.close(fig)
        
        # Visualize thermodynamic map
        fig = explainability_manager.visualize_thermodynamic(
            thermo_map, sequence_idx=0,
            save_path=os.path.join(VISUALIZATION_DIR, f'explanation_thermo_{i}.png')
        )
        plt.close(fig)
        
        # Visualize counterfactual explanation
        if counterfactual_explanation['counterfactuals']:
            fig = explainability_manager.visualize_counterfactual(
                counterfactual_explanation,
                save_path=os.path.join(VISUALIZATION_DIR, f'explanation_counterfactual_{i}.png')
            )
            plt.close(fig)
        
        # Visualize peptide graph
        fig = visualizer.visualize_peptide_graph(
            graph, sequence, residue_contributions,
            save_path=os.path.join(VISUALIZATION_DIR, f'explanation_graph_{i}.png')
        )
        plt.close(fig)
    
    # Generate global explanation
    global_explanation = explainability_manager.explain_global(
        all_sequences[:50], create_batch_graphs(all_sequences[:50]).to(device), method='permutation_importance'
    )
    
    fig = explainability_manager.visualize_global(
        global_explanation,
        save_path=os.path.join(VISUALIZATION_DIR, 'global_explanation.png')
    )
    plt.close(fig)
    
    print("Testing and visualization completed.")


def explain(args):
    """
    Generate explanations for the model.
    
    Args:
        args (argparse.Namespace): Command line arguments.
    """
    # Import create_batch_graphs at the beginning of the function
    from utils.data_processing import create_batch_graphs
    
    # Set device
    device = torch.device(args.device)
    
    # Load model
    print("Loading model...")
    model_path = args.model_path or os.path.join(MODEL_DIR, 'best_model.pth')
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model
    model = PeptideBindingModel(
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model from {model_path}")
    
    # Load data
    print("Loading data...")
    df = load_data(args.data_path)
    
    # Prepare datasets
    print("Preparing datasets...")
    _, _, test_dataset = prepare_datasets(df)
    
    # Initialize explainability manager
    explainability_manager = ExplainabilityManager(model)
    
    # Initialize visualizer
    visualizer = PeptideVisualizer(output_dir=VISUALIZATION_DIR)
    
    # Generate explanations for sample sequences
    num_samples = 10
    sample_indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    
    for i, idx in enumerate(sample_indices):
        sequence = test_dataset.sequences[idx]
        binding_energy = test_dataset.binding_energies[idx]
        
        # Create graph for the sequence
        graph = create_batch_graphs([sequence])[0].to(device)
        
        # Get model prediction
        with torch.no_grad():
            try:
                # Debug information
                print(f"Graph type: {type(graph)}")
                if hasattr(graph, 'x'):
                    print(f"Graph has x attribute with shape: {graph.x.shape}")
                
                # Pass the graph directly, not as a list
                outputs = model([sequence], graph)
                predicted_energy = outputs['binding_energy'].item()
            except Exception as e:
                print(f"Error during model prediction: {e}")
                print("Trying alternative approach...")
                
                # Create a new graph as a fallback
                from utils.data_processing import create_batch_graphs
                try:
                    # Create a new graph for the sequence
                    new_graph = create_batch_graphs([sequence]).to(device)
                    print(f"Created new graph with type: {type(new_graph)}")
                    
                    # Try prediction with the new graph
                    outputs = model([sequence], new_graph)
                    predicted_energy = outputs['binding_energy'].item()
                except Exception as e2:
                    print(f"Error with fallback approach: {e2}")
                    print("Using default value as last resort")
                    predicted_energy = 0.0  # Default value as last resort
        
        print(f"Sample {i+1}/{num_samples}:")
        print(f"Sequence: {sequence}")
        print(f"True Binding Energy: {binding_energy:.4f}")
        print(f"Predicted Binding Energy: {predicted_energy:.4f}")
        
        # Generate explanations
        print("Generating explanations...")
        
        # Local explanation
        for method in ['integrated_gradients', 'deep_lift', 'gradient_shap', 'occlusion']:
            local_explanation = explainability_manager.explain_local(
                [sequence], [graph], method=method
            )
            
            fig = explainability_manager.visualize_local(
                local_explanation,
                save_path=os.path.join(VISUALIZATION_DIR, f'sample_{i}_{method}.png')
            )
            plt.close(fig)
        
        # Thermodynamic map
        thermo_map = explainability_manager.explain_thermodynamic(
            [sequence], [graph]
        )
        
        fig = explainability_manager.visualize_thermodynamic(
            thermo_map, sequence_idx=0,
            save_path=os.path.join(VISUALIZATION_DIR, f'sample_{i}_thermo.png')
        )
        plt.close(fig)
        
        # Counterfactual explanation
        counterfactual_explanation = explainability_manager.explain_counterfactual(
            sequence, graph
        )
        
        if counterfactual_explanation['counterfactuals']:
            fig = explainability_manager.visualize_counterfactual(
                counterfactual_explanation,
                save_path=os.path.join(VISUALIZATION_DIR, f'sample_{i}_counterfactual.png')
            )
            plt.close(fig)
        
        print("Explanations generated.")
        print()
    
    # Generate global explanation
    print("Generating global explanation...")
    
    # Get a subset of sequences for global explanation
    # Ensure we don't try to sample more elements than are available
    subset_size = min(50, len(test_dataset))
    print(f"Using subset size of {subset_size} out of {len(test_dataset)} available samples")
    subset_indices = np.random.choice(len(test_dataset), subset_size, replace=False)
    subset_sequences = [test_dataset.sequences[idx] for idx in subset_indices]
    subset_graphs = create_batch_graphs(subset_sequences).to(device)
    
    # Generate global explanation
    global_explanation = explainability_manager.explain_global(
        subset_sequences, subset_graphs, method='permutation_importance'
    )
    
    fig = explainability_manager.visualize_global(
        global_explanation,
        save_path=os.path.join(VISUALIZATION_DIR, 'global_explanation.png')
    )
    plt.close(fig)
    
    print("Global explanation generated.")


def main():
    """
    Main function.
    """
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Run in the specified mode
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'explain':
        explain(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == '__main__':
    main()
