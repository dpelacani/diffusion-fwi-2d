import os
import json
import copy
import torch
import logging
import wandb
import matplotlib.pyplot as plt
from torch.amp import GradScaler
from tqdm import tqdm

from diffusionfwi.models import UNet
from diffusionfwi.dataset import build_dataset, get_dataloaders 
from diffusionfwi.diffusion import DiffusionProcess, train, valid
from diffusionfwi.utils import set_seed

from diff_experiments import BASE, EXPERIMENTS

def run_experiment(exp, seed):
    
    # ---------------------------- EXPERIMENT SETUP ----------------------------
    
    config = BASE | exp | {"seed": seed}
    run_name = f"{exp['name']}_seed{seed}"
    run_dir = os.path.join(BASE["work_dir"], run_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Set up logging for looping over experiments
    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(run_dir, "training.log"), mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    logger.info("Starting run: %s", run_name)

    # Save the config to a json file
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4, default=str, sort_keys=True)

    # Set random seed for reproducibility
    set_seed(seed)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ------------------------------ TRAINING SETUP ------------------------------
    
    # Build dataset
    train_dataset, val_dataset, test_dataset = build_dataset(
        BASE["data_dir"], true_model=BASE["true_model"], x_dim=BASE["x_dim"], 
        augment=exp["augment"], save_split_dir=run_dir
    )

    # Build dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size=BASE["batch_size"]
    )

    # Initialize the UNet model
    # in_channels is 2 for split (skull + soft tissue) channels, 1 otherwise
    num_channels = 2 if exp["split"] else 1
    model = UNet(
        in_channels=num_channels,
        out_channels=num_channels,
        time_emb_dim=BASE["time_emb_dim"],
        num_layers=exp["layers"],
        base_channels=exp["channels"]
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info("Model initialized with {:2E} parameters".format(num_params))

    # EMA model kept on CPU during training
    ema_model = copy.deepcopy(model).cpu()
    ema_model.eval()

    # Initialise Optimizer and loss function
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BASE["lr"],
        weight_decay=BASE["weight_decay"],
        betas=BASE["adam_betas"],
    )
    criterion = torch.nn.MSELoss()
    scaler = GradScaler("cuda")

    # Initialize diffusion process
    pipeline = DiffusionProcess(
        device=device, T=BASE["num_timesteps"], s=BASE["cosine_schedule_s"]
    )

    # Weights & Biases tracking (offline for cluster nodes, sync manually from login node)
    # Override with WANDB_MODE=online 
    wandb.init(
        project="diffusion-fwi-2d",
        name=run_name,
        config=config,
        mode=os.environ.get("WANDB_MODE", "offline"),
        dir=BASE["work_dir"]
    )

    # --------------------------------- TRAINING ---------------------------------
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    pbar = tqdm(range(BASE["num_epochs"]), desc=run_name, total=BASE["num_epochs"])
    for epoch in pbar:
        # Train the model
        train_loss = train(
            model,
            optimizer,
            criterion,
            train_loader,
            pipeline.forward_diffusion,
            scaler,
            device,
            T=BASE["num_timesteps"],
            ema_model=ema_model,
            split=exp["split"]
        )
        # Evaluate the model
        valid_loss = valid(
            model,
            criterion,
            val_loader,
            pipeline.forward_diffusion,
            device,
            T=BASE["num_timesteps"],
            split=exp["split"]
        )

        # Log losses
        train_losses.append(train_loss)
        val_losses.append(valid_loss)

        pbar.set_description(f"{run_name} | Train Loss: {train_loss:.4f}, Val Loss: {valid_loss:.4f}")
        logger.info("Epoch %d/%d - Train Loss: %.4f, Val Loss: %.4f", 
                    epoch+1, BASE["num_epochs"], train_loss, valid_loss)

        log_dict = {"train_loss": train_loss, "val_loss": valid_loss, "epoch": epoch+1}
        
        # Visualize EMA samples every 50 epochs
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                samples_viz = pipeline.sample_diffusion(
                    model, test_loader, batch_size=8, device=device, num_channels=num_channels
                )
            model.train()
            cmap = plt.get_cmap("terrain")
            imgs = (samples_viz[:8, 0].cpu().numpy() + 1) / 2
            log_dict["samples"] = [
                wandb.Image(cmap(img)[:, :, :3], caption=f"epoch{epoch+1}_{i}")
                for i, img in enumerate(imgs)
            ]

        wandb.log(log_dict)

        with open(os.path.join(run_dir, "losses.json"), "w") as f:
            json.dump({"train_losses": train_losses, "val_losses": val_losses}, f)
        
        # Save best validation checkpoint
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_val.pth"))

        # Save periodic checkpoints
        if (epoch + 1) % BASE["checkpoint_interval"] == 0 or (epoch+1) == BASE["num_epochs"]:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth"))

    # EMA checkpoint (used for model evaluation)
    ema_path = os.path.join(checkpoint_dir, "ema_final.pth")
    torch.save(ema_model.state_dict(), ema_path) 
    logger.info("Saved EMA checkpoint: %s", ema_path)

    wandb.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None,
                        help="Run a specific experiment by name (default: all)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Run a specific seed (default: all seeds in BASE)")
    args = parser.parse_args()

    exps  = [e for e in EXPERIMENTS if args.name is None or e["name"] == args.name]
    seeds = [args.seed] if args.seed is not None else BASE["seeds"]

    for exp in exps: 
        for seed in seeds:
            run_experiment(exp, seed)