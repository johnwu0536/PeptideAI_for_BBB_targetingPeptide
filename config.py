"""
Configuration settings for the PeptideAI project.
"""

import os

# General settings
RANDOM_SEED = 42
DATA_PATH = os.path.join('data', 'Peptide.csv')
OUTPUT_DIR = os.path.join('outputs')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, 'visualizations')

# Data settings
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEQ_MAX_LENGTH = 50

# Amino acid properties
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_PROPERTIES = {
    'hydrophobicity': {
        'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
        'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
        'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
        'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
    },
    'charge': {
        'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0,
        'G': 0, 'H': 0.1, 'I': 0, 'K': 1, 'L': 0,
        'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1,
        'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0
    },
    'weight': {
        'A': 89.09, 'C': 121.16, 'D': 133.10, 'E': 147.13, 'F': 165.19,
        'G': 75.07, 'H': 155.16, 'I': 131.17, 'K': 146.19, 'L': 131.17,
        'M': 149.21, 'N': 132.12, 'P': 115.13, 'Q': 146.15, 'R': 174.20,
        'S': 105.09, 'T': 119.12, 'V': 117.15, 'W': 204.23, 'Y': 181.19
    }
}

# Model settings
# Sequence encoder
SEQ_EMBEDDING_DIM = 256
SEQ_NUM_HEADS = 8
SEQ_NUM_LAYERS = 6
SEQ_DROPOUT = 0.1

# Graph neural network
GNN_HIDDEN_DIM = 256
GNN_NUM_LAYERS = 3
GNN_DROPOUT = 0.1

# Cross-modal attention
CROSS_MODAL_HEADS = 8
CROSS_MODAL_DROPOUT = 0.1

# Training settings
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
GRADIENT_CLIP_VAL = 1.0

# Dynamic optimization settings
FEEDBACK_INTERVAL = 5
CONTRADICTION_THRESHOLD = 0.7

# Explainability settings
COUNTERFACTUAL_NUM_SAMPLES = 100
