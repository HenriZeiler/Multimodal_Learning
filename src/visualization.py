"""
Visualization utilities for multimodal learning experiments.

This module provides functions for visualizing training progress,
model performance, and data distributions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os


def _ensure_save_dir(save_path):
    """
    Helper function to ensure the directory for save_path exists.
    
    Args:
        save_path: Path where a file will be saved
    """
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:  # Only create if there's actually a directory component
            os.makedirs(save_dir, exist_ok=True)


def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Dictionary containing training history
        save_path: Optional path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        _ensure_save_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Optional path to save the figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    if save_path:
        _ensure_save_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_model_comparison(results, metric='accuracy', save_path=None):
    """
    Plot comparison of different models.
    
    Args:
        results: Dictionary with model names as keys and metrics as values
        metric: Metric to compare (e.g., 'accuracy', 'f1_score')
        save_path: Optional path to save the figure
    """
    models = list(results.keys())
    values = [results[model][metric] for model in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, values, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}%', ha='center', va='bottom')
    
    plt.xlabel('Model')
    plt.ylabel(f'{metric.capitalize()} (%)')
    plt.title(f'Model Comparison - {metric.capitalize()}')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        _ensure_save_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_data_distribution(data, labels, label_names=None, save_path=None):
    """
    Plot data distribution for different classes.
    
    Args:
        data: Data samples
        labels: Labels for the data
        label_names: Names of the labels
        save_path: Optional path to save the figure
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    
    if label_names is None:
        label_names = [f'Class {i}' for i in unique_labels]
    
    bars = plt.bar(label_names, counts, color='lightcoral', edgecolor='darkred', alpha=0.7)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom')
    
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('Data Distribution Across Classes')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        _ensure_save_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def print_classification_report(y_true, y_pred, class_names=None):
    """
    Print classification report with precision, recall, and F1-score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    print("\nClassification Report:")
    print("=" * 60)
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
