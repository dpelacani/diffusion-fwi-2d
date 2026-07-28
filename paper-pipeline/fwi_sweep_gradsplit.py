import os
import torch
from stride import *

from fwi_experiments import BASE_FWI, EXPERIMENTS, DIFFUSION_MODELS
from diffusionfwi.diffusion import DiffusionProcess
from diffusionfwi.models import UNet
from diffusionfwi.fwi import ScoreGuidanceOperator

def _load_model(model_str, device):
    """ Load EMA checkpoint and return diffusion model in eval state. """
    cfg = DIFFUSION_MODELS[model_str]
    num_channels = 2 if cfg["split"] else 1
    model = UNet(
        in_channels=num_channels, out_channels=num_channels,
        time_emb_dim=256, num_layers=cfg["layers"], base_channels=cfg["channels"]
    )
    model.load_state_dict(torch.load(cfg["checkpoint"], map_location="cpu"))
    model.eval()
    return model.to(device)


async def main(runtime):

    name            = os.environ.get("MODEL_NAME",          "vp_996782")
    experiment_name = os.environ.get("SWEEP_EXPERIMENT",    "reduced_compute")
    t_start         = int(os.environ.get("SWEEP_T_START",   "700"))
    t_end           = int(os.environ.get("SWEEP_T_END",     "100"))
    lambda_skull    = float(os.environ.get("SWEEP_LAMBDA_SKULL",  "1.0"))
    lambda_tissue   = float(os.environ.get("SWEEP_LAMBDA_TISSUE", "0.5"))
    lambda_end      = float(os.environ.get("SWEEP_LAMBDA_END",    "0.01"))
    warmstart_iters = int(os.environ.get("SWEEP_WARMSTART_ITERS", "8"))

    sweep_base = os.environ.get("SWEEP_OUT_DIR", BASE_FWI["sweep_output_dir"])

    exp = EXPERIMENTS[experiment_name]

    skull_tag  = int(round(lambda_skull  * 100))
    tissue_tag = int(round(lambda_tissue * 100))
    output_dir = os.path.join(
        sweep_base,
        f"sg_c_{name}_{experiment_name}_t{t_start}-{t_end}"
        f"_ls{skull_tag:03d}_lt{tissue_tag:03d}_ws{warmstart_iters}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Skip if already done
    done_file = os.path.join(output_dir, "done")
    if os.path.exists(done_file):
        print(f"[skip] {output_dir} already done")
        return

    print(f"=== SG-C (score guidance, per-channel) sweep ===")
    print(f"  brain:           {name}")
    print(f"  experiment:      {experiment_name}")
    print(f"  t_start/t_end:   {t_start} -> {t_end}")
    print(f"  lambda_skull:    {lambda_skull}  -> {lambda_end}")
    print(f"  lambda_tissue:   {lambda_tissue} -> {lambda_end}")
    print(f"  warmstart_iters: {warmstart_iters}")
    print(f"  block_iters:     {exp['block_iters']}  (total {exp['total_iters']})")
    print(f"  max_freqs (kHz): {[f/1e3 for f in exp['max_freqs']]}")
    print(f"  output:          {output_dir}")

    input_dir = os.path.join(BASE_FWI["input_dir"], name.upper())

    space = Space(
        shape=BASE_FWI["shape"], extra=BASE_FWI["extra"],
        absorbing=BASE_FWI["absorbing"], spacing=BASE_FWI["spacing"]
    )
    time = Time(start=0.0, step=BASE_FWI["time_step"], num=BASE_FWI["num_steps"])

    problem = Problem(
        name=name, space=space, time=time,
        input_folder=input_dir, output_folder=output_dir
    )
    problem.output_folder = output_dir   # redirect all Stride file writes here

    vp = ScalarField.parameter(name="vp", grid=problem.grid, needs_grad=True)
    vp.fill(1500.0)
    problem.medium.add(vp)

    problem.transducers.default()
    problem.geometry.default(
        "elliptical", BASE_FWI["num_transducers"],
        radius=(
            (space.limit[0] - 7.0e-3) / 2,
            (space.limit[1] - 5.0e-3) / 2,
        )
    )
    problem.acquisitions.load(
        filename=os.path.join(input_dir, f"{name.upper()}-Acquisitions.h5")
    )

    pde  = IsoAcousticDevito.remote(grid=problem.grid, len=runtime.num_workers)
    loss = L2DistanceLoss.remote(len=runtime.num_workers)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    process_grad  = ProcessGlobalGradient()
    process_model = ProcessModelIteration(min=1480.0, max=3000.0)

    operator = ScoreGuidanceOperator(
        diffusion_model    = _load_model("reference", device),
        diffusion_process  = DiffusionProcess(T=BASE_FWI["T"], s=BASE_FWI["s"], device=device),
        vp_ref             = vp,
        input_dim          = BASE_FWI["input_dim"],
        original_dim       = BASE_FWI["shape"],
        split              = True,
        max_freqs          = exp["max_freqs"],
        t_start            = t_start,
        t_end              = t_end,
        lambda_skull       = lambda_skull,
        lambda_tissue      = lambda_tissue,
        lambda_end         = lambda_end,
        warmstart_iters    = warmstart_iters,
        device             = device,
        visual_dir         = output_dir
    )
    process_grad.append("diffusion", operator)

    optimiser = GradientDescent(
        vp,
        step_size     = BASE_FWI["step_size"],
        process_grad  = process_grad,
        process_model = process_model,
        dump_grad     = False,
    )
    optimisation_loop = OptimisationLoop()

    for i, (block, freq) in enumerate(
            optimisation_loop.blocks(len(exp["max_freqs"]), exp["max_freqs"])):
        num_iters = exp["block_iters"][i]
        for iteration in block.iterations(num_iters):
            await adjoint(
                problem, pde, loss, optimisation_loop, optimiser, vp,
                num_iters   = num_iters,
                select_shots= dict(num=BASE_FWI["num_shots"],
                                   every=BASE_FWI["every"], randomly=False),
                f_max       = freq,
                max_freqs   = exp["max_freqs"],
                kernel      = "OT4",
                platform    = "nvidia-acc",
            )
            optimiser.dump(
                path         = output_dir,
                project_name = problem.name,
                version      = iteration.abs_id + 1,
            )

    print(f"[done] {name}  experiment={experiment_name}  ls={lambda_skull}  lt={lambda_tissue}")
    open(done_file, "w").close()


if __name__ == "__main__":
    mosaic.run(main, log_level="perf")