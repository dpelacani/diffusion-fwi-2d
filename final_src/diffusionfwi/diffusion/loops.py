import torch

from diffusionfwi.utils import split_channels

EMA_DECAY = 0.9993

def train(model, optimizer, criterion, data_loader, forward_diffusion, scaler, device, T=1000, split=True, ema_model=None):
    """
    Train the diffusion model.

    Args:
        model: model to train (UNet).
        optimizer: optimizer for training.
        criterion: loss function to use.
        data_loader: data loader for the training data.
        forward_diffusion: forward diffusion process function.
        scaler: GradScaler instance for mixed-precision training.
        device: device to run the computations on.
        T: number of timesteps for the diffusion process.
        split: if True, split single-channel input into skull/soft-tissue channels before passing to model.
        ema_model: EMA model updated in-place each training step. if None, EMA updating is skipped.

    Returns:
        The average loss for the training epoch.
    """
    model.train()
    total_loss = 0.0

    for x0 in data_loader:
        x0 = x0.to(device)
        x0 = split_channels(x0) if split else x0

        # Sample time steps t ∈ [0, T)
        t = torch.randint(0, T, (x0.size(0),), dtype=torch.long).to(device)

        # Sample noise e from N(0, I)
        e = torch.randn_like(x0)

        # Corrupt x0 into xt using forward diffusion process
        xt = forward_diffusion(x0, t, e)

        # Autocast to mixed precision for faster training and reduced memory usage
        with torch.amp.autocast("cuda"):
            # Predict the noise using the model
            pred_e = model(xt, t)
            
            # Compute loss
            loss = criterion(pred_e, e)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update() 
    
        if ema_model is not None:
            with torch.no_grad():
                for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                    ema_param.data.mul_(EMA_DECAY).add_(param.data.cpu(), alpha=1-EMA_DECAY)

        # Accumulate loss for the epoch
        total_loss += loss.item() * x0.size(0)

    avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss

def valid(model, criterion, data_loader, forward_diffusion, device, T=1000, split=True):
    """
    Validation loop for diffusion model.

    Args:
        model: noise prediction model
        criterion: loss function (typically MSELoss)
        data_loader: validation data loader
        forward_diffusion: function to compute xt from x0, t, e
        T: max diffusion step
        split: if True, split single-channel input into skull/soft-tissue channels before passing to model. 

    Returns:
        The avg validation loss over the entire validation set.
    """
    model.eval()
    valid_loss = 0.0

    with torch.no_grad():
        for x0 in data_loader:
            x0 = (x0[0] if isinstance(x0, (list, tuple)) else x0).to(device)
            x0 = split_channels(x0) if split else x0

            # Sample random time steps
            t = torch.randint(0, T, (x0.size(0),), dtype=torch.long).to(device)

            # Sample Gaussian noise
            e = torch.randn_like(x0)

            # Get noisy sample
            xt = forward_diffusion(x0, t, e)

            # Autocast to mixed precision for faster validation and reduced memory usage
            with torch.amp.autocast("cuda"):
                # Predict noise
                pred_e = model(xt, t)

                # Compute loss
                loss = criterion(pred_e, e)

            # Accumulate loss for the validation set
            valid_loss += loss.item() * x0.size(0)

    # Compute average validation loss
    avg_valid_loss = valid_loss / len(data_loader.dataset)
    return avg_valid_loss