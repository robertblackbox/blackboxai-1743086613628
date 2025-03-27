import json
import logging
import os
from typing import Dict, Any, Optional
import torch
import matplotlib.pyplot as plt


def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file (Optional[str]): Path to log file. If None, logs to console only.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger('neural_network')
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path (str): Path to config JSON file
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in config file: {config_path}")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model (torch.nn.Module): Neural network model
        optimizer (torch.optim.Optimizer): Optimizer instance
        epoch (int): Current epoch number
        loss (float): Current loss value
        path (str): Path to save checkpoint
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        path (str): Path to checkpoint file
        model (torch.nn.Module): Neural network model
        optimizer (Optional[torch.optim.Optimizer]): Optimizer instance
    
    Returns:
        Dict[str, Any]: Checkpoint data including epoch and loss
    
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at: {path}")
    
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'epoch': checkpoint['epoch'],
        'loss': checkpoint['loss']
    }


def plot_training_history(
    train_losses: list,
    val_losses: list,
    save_path: Optional[str] = None
) -> None:
    """
    Plot training and validation loss history.
    
    Args:
        train_losses (list): List of training losses
        val_losses (list): List of validation losses
        save_path (Optional[str]): Path to save the plot. If None, displays the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def get_device() -> torch.device:
    """
    Get the available device (CPU/GPU) for training.
    
    Returns:
        torch.device: Available device
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')