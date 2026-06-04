import os
import argparse
import torch
from stride import *

from fwi_experiments import BASE_FWI, SCENARIOS, METHODS, DIFFUSION_MODELS
from diffusionfwi.diffusion import DiffusionProcess
from diffusionfwi.models import UNet
from diffusionfwi.fwi import PostprocessingOperator, GradientOperator

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

# ---------------------------- SINGLE RUN ----------------------------

async def _run_inversion(runtime, problem, pde, loss, vp, brain, scenario, method, device):
    """
    Run one experiment (scenario + method) for the given brain.

    - problem, pde, loss, vp are shared objectes created in main() (vp reset for each run)
    - process_grad, process_model and optimization setup is created for each inversion run
    """
    sc = SCENARIOS[scenario]
    m = METHODS[method]

    # Output directory for inversion run
    output_dir = os.path.join(BASE_FWI["output_dir"], f"{brain}_{scenario}_{method}")
    os.makedirs(output_dir, exist_ok=True)
    problem.output_folder = output_dir

    print(f"\n--- FWI inversion run: brain={brain} scenario={scenario} method={method} ---")
    print(f"block iters: {sc['block_iters']} (total {sc['total_iters']})")
    print(f"max_freqs: {[f / 1e3 for f in sc['max_freqs']]} kHz")
    print(f"guidance: {m['guidance']}")

    # Reset velocity model
    vp.fill(1500.0)

    # -------------------- SETUP DIFFUSION GUIDANCE --------------------

    process_grad = ProcessGlobalGradient()
    process_model = ProcessModelIteration(min=1480.0, max=3000.0)

    if m["guidance"] == "postprocessing":
        operator = PostprocessingOperator(
            diffusion_model   = _load_model(m["diffusion_model"], device),
            diffusion_process = DiffusionProcess(T=BASE_FWI["T"], s=BASE_FWI["s"], device=device),
            iters_to_run      = sc["iters_to_run"],
            t_start           = m["t_start"],
            t_end             = m["t_end"],
            alpha_skull       = m["alpha_skull"],
            alpha_tissue      = m["alpha_tissue"],
            alpha_end         = m["alpha_end"],
            split             = m["split"],
            input_dim         = BASE_FWI["input_dim"],
            original_dim      = BASE_FWI["shape"],
            device            = device
        )
        process_model.append("diffusion", operator)
    
    elif m["guidance"] == "gradient":
        operator = GradientOperator(
            diffusion_model   = _load_model(m["diffusion_model"], device),
            diffusion_process = DiffusionProcess(T=BASE_FWI["T"], s=BASE_FWI["s"], device=device),
            vp_ref            = vp,
            max_freqs         = sc["max_freqs"],
            t_start           = m["t_start"],
            t_end             = m["t_end"],
            lambda_skull      = m["lambda_skull"],
            lambda_tissue     = m["lambda_tissue"],
            lambda_end        = m["lambda_end"],
            split             = m["split"],
            input_dim         = BASE_FWI["input_dim"],
            original_dim      = BASE_FWI["shape"],
            device            = device
        )
        process_grad.append("diffusion", operator)

    # Baseline case: no operator appended, plain FWI with no diffusion guidance
    
    # ---------------------------- FWI LOOP ----------------------------
    optimiser = GradientDescent(
        vp,
        step_size = BASE_FWI["step_size"],
        process_grad = process_grad,
        process_model = process_model,
        dump_grad = False
    )
    optimisation_loop = OptimisationLoop()
    num_blocks = len(sc["max_freqs"])

    for i, (block, freq) in enumerate(optimisation_loop.blocks(num_blocks, sc["max_freqs"])):
        num_iters = sc["block_iters"][i]
        for iteration in block.iterations(num_iters):
            await adjoint(
                problem, pde, loss, optimisation_loop, optimiser, vp,
                num_iters=num_iters,
                select_shots=dict(
                    num=BASE_FWI["num_shots"], 
                    every=BASE_FWI["every"], 
                    randomly=False
                ),    
                f_max=freq,
                max_freqs=sc["max_freqs"],
                kernel="OT4",
                platform="nvidia-acc",
            )
            optimiser.dump(
                path=output_dir,
                project_name=problem.name,
                version=iteration.abs_id + 1,
            )

    print(f"[done] {brain}  {scenario}  {method}")

async def main(runtime):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n--- FWI inversion run: brain={brain} device={device} ---")
    print(f"scenarios: {scenarios}")
    print(f"methods: {methods}")

    input_dir = os.path.join(BASE_FWI["input_dir"], brain.upper())
    
    # --------------------------- STRIDE SETUP ---------------------------
    # For specific brain: space, time, grid, transducer geometry, acquisition data

    space = Space(
        shape = BASE_FWI["shape"],
        extra = BASE_FWI["extra"],
        absorbing = BASE_FWI["absorbing"],
        spacing = BASE_FWI["spacing"]
    )
    time = Time(start=0.0, step=BASE_FWI["time_step"], num=BASE_FWI["num_steps"])
    problem = Problem(
        name          = brain,
        space         = space,
        time          = time,
        input_folder  = input_dir,
        output_folder = BASE_FWI["output_dir"]
    )

    vp = ScalarField.parameter(name="vp", grid=problem.grid, needs_grad=True)
    vp.fill(1500.0)
    problem.medium.add(vp)

    problem.transducers.default()
    problem.geometry.default(
        "elliptical", BASE_FWI["num_transducers"],
        radius=(
            (space.limit[0] - 7.0e-3)/2, 
            (space.limit[1] - 5.0e-3)/2
        )
    )
    problem.acquisitions.load(
        filename=os.path.join(input_dir, f"{brain.upper()}-Acquisitions.h5")
    )

    # PDE solver and loss
    pde  = IsoAcousticDevito.remote(grid=problem.grid, len=runtime.num_workers)
    loss = L2DistanceLoss.remote(len=runtime.num_workers)

    for scenario in scenarios:
        for method in methods:
            await _run_inversion(
                runtime, problem, pde, loss, vp,
                brain, scenario, method, device
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion-guided FWI")
    parser.add_argument("--brain", default=os.environ.get("BRAIN", "vp_996782"))
    parser.add_argument("--scenario", default=os.environ.get("SCENARIO", None))
    parser.add_argument("--method", default=os.environ.get("METHOD", None))
    args, _ = parser.parse_known_args()

    brain = args.brain
    scenarios = [args.scenario] if args.scenario else list(SCENARIOS.keys())
    methods = [args.method] if args.method else list(METHODS.keys())

    for sc in scenarios:
        if sc not in SCENARIOS:
            raise ValueError(f"Unknown scenario {sc}. Scenarios = {list(SCENARIOS.keys())}")
        
    for mt in methods:
        if mt not in METHODS:
            raise ValueError(f"Unknown method {mt}. Methods = {list(METHODS.keys())}")
    
    mosaic.run(main, log_level="perf")