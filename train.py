import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import json
import os
from typing import Tuple, List

from model import NeuralNet
from utils import (
    setup_logging,
    load_config,
    save_checkpoint,
    load_checkpoint,
    plot_training_history,
    get_device
)


def prepare_data(config: dict) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare MNIST dataset and create data loaders.
    
    Args:
        config (dict): Configuration dictionary
    
    Returns:
        Tuple[DataLoader, DataLoader]: Train and validation data loaders
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Load MNIST dataset
    dataset = datasets.MNIST(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )
    
    # Split into train and validation sets
    train_size = int(len(dataset) * config['data_params']['train_split'])
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config['data_params']['random_seed'])
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training_params']['batch_size'],
        shuffle=config['data_params']['shuffle']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training_params']['batch_size'],
        shuffle=False
    )
    
    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    logger: logging.Logger
) -> float:
    """
    Train the model for one epoch.
    
    Args:
        model (nn.Module): Neural network model
        train_loader (DataLoader): Training data loader
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to train on
        logger (logging.Logger): Logger instance
    
    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Flatten the input (MNIST images are 28x28)
        data = data.view(data.size(0), -1)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            logger.info(f'Training batch {batch_idx}/{len(train_loader)}, '
                       f'Loss: {loss.item():.6f}')
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Validate the model.
    
    Args:
        model (nn.Module): Neural network model
        val_loader (DataLoader): Validation data loader
        criterion (nn.Module): Loss function
        device (torch.device): Device to validate on
    
    Returns:
        float: Average validation loss
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def main():
    """Main training function."""
    # Set up logging
    logger = setup_logging('training.log')
    
    try:
        # Load configuration
        config = load_config('config.json')
        
        # Set device
        device = get_device()
        logger.info(f'Using device: {device}')
        
        # Prepare data
        train_loader, val_loader = prepare_data(config)
        logger.info('Data loaders created successfully')
        
        # Initialize model
        model = NeuralNet(
            input_size=config['model_params']['input_size'],
            hidden_sizes=config['model_params']['hidden_sizes'],
            output_size=config['model_params']['output_size'],
            dropout_rate=config['model_params']['dropout_rate']
        ).to(device)
        
        logger.info(f'Model created with {model.get_num_parameters()} parameters')
        
        # Initialize criterion and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training_params']['learning_rate']
        )
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        for epoch in range(config['training_params']['epochs']):
            logger.info(f'Starting epoch {epoch + 1}')
            
            # Train
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, device, logger
            )
            train_losses.append(train_loss)
            
            # Validate
            val_loss = validate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            
            logger.info(
                f'Epoch {epoch + 1}: '
                f'Training Loss = {train_loss:.6f}, '
                f'Validation Loss = {val_loss:.6f}'
            )
            
            # Save checkpoint if best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    val_loss,
                    config['training_params']['model_save_path']
                )
                logger.info('Saved new best model checkpoint')
        
        # Plot training history
        plot_training_history(
            train_losses,
            val_losses,
            save_path='training_history.png'
        )
        logger.info('Training completed successfully')
        
    except Exception as e:
        logger.error(f'Error during training: {str(e)}', exc_info=True)
        raise


if __name__ == '__main__':
    main()