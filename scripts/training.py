import os
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import logging

from diffusionfwi.models import UNetAttnMoreD
from diffusionfwi.dataset import build_dataset, get_dataloaders
from diffusionfwi.utils import *
from diffusionfwi.diffusion import DiffusionProcess
from diffusionfwi.loops import train, valid
from diffusionfwi.utils import plot_batch, plot_losses

from datetime import datetime

CONFIG = {
    "experiment_name": "diffusion_fwi_2d",
    "work_dir": "./exps/",
    "data_dir": "/scratch_hive/dp4018/data/ultrasound-data/Ultrasound-Vp-axial-models/",
    "true_model": "vp_996782.npy",
    "x_dim": 8,
    "batch_size": 16,
    "num_timesteps": 1000,
    "cosine_schedule_s": 0.001,
    "model_params":{
        "time_emb_dim": 256,
        "in_channels": 1,
        "out_channels": 1,
    },
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "adam_betas": (0.9, 0.98),
    "num_epochs": 2,
    "checkpoint_interval": 25,
    "seed": 42,
    "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
}

if __name__ == "__main__":

    # EXPERIMENT SETUP
    #############################################################################

    # Experiment configuration
    experiment_name = CONFIG["experiment_name"]
    experiment_dir = f"{CONFIG['work_dir']}{experiment_name}"
    logging_dir = f"{experiment_dir}/logs"
    checkpoint_dir = f"{experiment_dir}/checkpoints"

    # Create directories for the experiment
    os.makedirs(logging_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Set up logging
    logging.basicConfig(
        filename=f"{logging_dir}/training.log",
        filemode='w',
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Starting experiment: %s", experiment_name)

    # Set random seed for reproducibility
    set_seed(CONFIG["seed"])

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)


    # TRAINING SETUP
    #############################################################################

    # Build dataset
    x_dim = CONFIG["x_dim"]
    true_model = CONFIG["true_model"]
    data_dir = CONFIG["data_dir"]
    train_dataset, val_dataset, test_dataset = build_dataset(data_dir, true_model=true_model, x_dim=x_dim)

    # Build dataloaders
    batch_size = CONFIG["batch_size"]
    train_loader, val_loader, test_loader = get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=batch_size)

    # Initialize model and diffusion process
    pipeline = DiffusionProcess(device=device, T=CONFIG["num_timesteps"], s=CONFIG["cosine_schedule_s"])

    # Plot batch of samples
    batch = next(iter(train_loader))[:16]
    plot_batch(batch, nrow=8, title="Batch of Training Samples", save_path=f"{logging_dir}/train_samples.png")

    # Plot batch of corrupted samples at different timesteps
    random_Ts = random.sample(range(CONFIG["num_timesteps"]), len(train_dataset))
    t = torch.randint(0, CONFIG["num_timesteps"], (batch.size(0),), dtype=torch.long).to(device)
    e = torch.randn_like(batch[0])
    xt = pipeline.forward_diffusion(batch, t, e)
    plot_batch(xt[:16], nrow=8, title="Batch of Corrupted Samples at Random Timesteps", save_path=f"{logging_dir}/corrupted_samples.png")

    # Initialize the UNet model
    model = UNetAttnMoreD(**CONFIG["model_params"]).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logging.info("Model initialized with %d parameters", num_params)

    # Initialise Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"],
        betas=CONFIG["adam_betas"]
    )
    criterion = torch.nn.MSELoss()

    # Add model, optimiser, and criterion to the config json file
    CONFIG["model"] = model.__class__.__name__
    CONFIG["num_params"] = num_params
    CONFIG["model_config"] = {"type": str(model.__class__)} 
    CONFIG["optimizer"] = optimizer.__class__.__name__
    CONFIG["optimizer_config"] = optimizer.defaults
    CONFIG["criterion"] = criterion.__class__.__name__

    # Save the config to a json file
    with open(f"{experiment_dir}/config.json", "w") as f:
        json.dump(CONFIG, f, indent=4, default=str, sort_keys=True)


    # TRAINING
    #############################################################################

    # Train and validate the model
    train_losses, val_losses = [], []
    for epoch in range(CONFIG["num_epochs"]):
        logging.info("Epoch %d/%d", epoch + 1, CONFIG["num_epochs"])

        # Train the model
        train_loss = train(model, optimizer, criterion, train_loader, pipeline.forward_diffusion, device, T=CONFIG["num_timesteps"])
        
        # Evaluate the model
        valid_loss = valid(model, criterion, val_loader, pipeline.forward_diffusion, device, T=CONFIG["num_timesteps"])

        # Log the losses
        logging.info("Train Loss: %.4f, Val Loss: %.4f", train_loss, valid_loss)
        train_losses.append(train_loss)
        val_losses.append(valid_loss)

        # Dump the losses into a json file and plot it
        with open(f"{logging_dir}/losses.json", "w") as f:
            json.dump({"train_losses": train_losses, "val_losses": val_losses}, f)
        plot_losses(train_losses, val_losses, save_path=f"{logging_dir}/loss_plot.png")


        # Save model checkpoint
        if (epoch + 1) % CONFIG["checkpoint_interval"] == 0 or (epoch + 1) == CONFIG["num_epochs"]:
            checkpoint_path = f"{checkpoint_dir}/model_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            logging.info("Saved model checkpoint at: %s", checkpoint_path)


    # EVALUATION & METRICS
    #############################################################################
    
    # Sample one random batch
    samples = pipeline.sample_diffusion(model, test_loader, batch_size=CONFIG["batch_size"], device=device)
    np.save(f"{experiment_dir}/sampled_images.npy", samples.cpu().numpy())
    logging.info("Saved sampled images at: %s", f"{experiment_dir}/sampled_images.npy")

    # Plot samples
    plot_batch(samples[:16], nrow=8, title="Sampled Images", save_path=f"{logging_dir}/sampled_images.png")

    # Evaluate loss on test set
    test_loss = valid(model, criterion, test_loader, pipeline.forward_diffusion, device, T=CONFIG["num_timesteps"])
    logging.info("Test Loss: %.4f", test_loss)

    # # Evaluation metrics
    # rad_model = RadImageNetFeaturesFID(pretrained_model)
    # metrics = EvaluationMetrics(next(iter(test_loader)), samples, rad_model, device)
    # metrics.plot_hist(save_path=f"{logging_dir}/histogram.png")
    # metrics.plot_hist_soft_tissue(save_path=f"{logging_dir}/histogram_soft_tissue.png")
    # real_stats = metrics.statistics(next(iter(test_loader)))
    # sample_stats = metrics.statistics(samples)
    # metrics_vals = {
    #     "real_mean": real_stats[0],
    #     "real_std": real_stats[1],
    #     "real_min": real_stats[2],
    #     "real_max": real_stats[3],
    #     "sample_mean": sample_stats[0],
    #     "sample_std": sample_stats[1],
    #     "sample_min": sample_stats[2],
    #     "sample_max": sample_stats[3],
    #     "fid_rad": metrics.fid(use_radinet=True, model=rad_model),
    #     "fid": metrics.fid(use_radinet=False),
    #     "mmd": metrics.mmd(),
    #     "mmd_feat": metrics.mmd(feat=True),
    #     "lpips": metrics.lpips(samples),
    # }
    # logging.info("Evaluation Metrics: %s", metrics_vals)
    # with open(f"{logging_dir}/metrics.json", "w") as f:
    #     json.dump(metrics_vals, f)


