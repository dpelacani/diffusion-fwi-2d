from tqdm import tqdm
import torch

# Define the validation loop
def valid(model, criterion, data_loader, forward_diffusion, device, T=1000):
    """
    Validation loop for diffusion model.

    Args:
    Parameters:
    - model: the noise prediction model
    - criterion: loss function (typically MSELoss)
    - data_loader: validation data loader
    - forward_diffusion: function to compute xt from x0, t, e
    - T: max diffusion step

    Returns:
    - avg validation loss over the entire validation set
    """
    model.eval()
    valid_loss = 0.0

    with torch.no_grad():
        for x0 in data_loader:
            x0 = x0[0].to(device)

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

            valid_loss += loss.item() * x0.size(0)

    avg_valid_loss = valid_loss / len(data_loader.dataset)
    # clean up GPU memory to prevent out of memory error
    torch.cuda.empty_cache()
    return avg_valid_loss