from torch.utils.data import DataLoader


def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=16):
    """
    Create DataLoaders for training, validation, and testing datasets.

    Args:
        train_dataset: training dataset.
        val_dataset: validation dataset.
        test_dataset: test dataset.
        batch_size: batch size for the DataLoaders.

    Returns:
        A tuple containing the training, validation, and test DataLoaders.
    """
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,     
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    return train_loader, val_loader, test_loader