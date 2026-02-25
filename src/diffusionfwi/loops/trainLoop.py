from tqdm import tqdm
import torch

# Define the training loop
def train(model, optimizer, criterion, data_loader, forward_diffusion, device, T=1000):
    """
    Train the diffusion model.

    Args:
        model: The model to train (UNet).
        optimizer: The optimizer for training.
        criterion: The loss function to use.
        data_loader: The data loader for the training data.
        forward_diffusion: The forward diffusion process function.
        device: The device to run the computations on.
        T: The number of timesteps for the diffusion process.

    Returns:
        The average loss for the training epoch.
    """
    model.train()
    total_loss = 0.0

    pbar = tqdm(data_loader)
    
    for x0 in pbar:
        x0 = x0.to(device)

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
        loss.backward()
        optimizer.step()

        # Update progress bar and accumulate loss
        pbar.set_description(f"loss: {loss.item():.4f}")
        total_loss += loss.item() * x0.size(0)

    avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss