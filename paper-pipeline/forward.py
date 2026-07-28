import os
import numpy as np
import argparse
import logging
from stride import *
from stride.utils import wavelets

from fwi_experiments import BASE_FWI
from diffusionfwi.fwi.utils import npy2h5

logger = logging.getLogger(__name__)

async def main(runtime):
    # Create the spatio-temporal grid
    space = Space(
        shape     = BASE_FWI["shape"], 
        extra     = BASE_FWI["extra"], 
        absorbing = BASE_FWI["absorbing"], 
        spacing   = BASE_FWI["spacing"]
    )

    time = Time(
        start = 0.0, 
        step  = BASE_FWI["time_step"], 
        num   = BASE_FWI["num_steps"]
    )

    name = brain.upper()
    experiment_dir = os.path.join(BASE_FWI["input_dir"], name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Create h5 from npy true model
    h5_path = os.path.join(experiment_dir, f"{name}.h5")
    if not os.path.isfile(h5_path):
        logger.info("Converting %s to HDF5...", f"{brain}.npy")
        npy2h5(
            path      = os.path.join(BASE_FWI["data_dir"], f"{brain}.npy"),
            name      = f"{name}.h5",
            save_path = experiment_dir,
            t_num     = BASE_FWI["num_steps"],
            t_step    = BASE_FWI["time_step"],
            t_start   = 0.0,
            extra     = BASE_FWI["extra"],
            absorbing = BASE_FWI["absorbing"],
            spacing   = BASE_FWI["spacing"],
        )  
    logger.info(f"Created HDF5 file for true model at {experiment_dir}/{name}.h5")

    # --------------- Problem setup ---------------
    problem = Problem(
        name          = name,
        input_folder  = experiment_dir,
        output_folder = experiment_dir,
        space         = space,
        time          = time
    )

    # Create medium (speed of sound of the region of interest)
    # vp contains a numpy array with the velocity for every point on the grid
    vp = ScalarField(name="vp", grid=problem.grid)
    vp.load(h5_path)
    problem.medium.add(vp)

    # Create transducers (assume point transducers in simulation)
    problem.transducers.default()

    # Create geometry (create ring of 256 transducers around the region of interest)
    problem.geometry.default(
        "elliptical",
        BASE_FWI["num_transducers"],
        radius=(
            (space.limit[0] - 7.0e-3) / 2, 
            (space.limit[1] - 5.0e-3) / 2
        )
    )

    # Create acquisitions (link every transducer in the region of interest with each other)
    problem.acquisitions.default()

    # Create wavelets (tone bursts)
    f_centre = 0.25e6 # Hz (center frequency)
    n_cycles = 3

    for shot in problem.acquisitions.shots:
        shot.wavelets.data[0, :] = wavelets.tone_burst(
            f_centre, n_cycles, time.num, time.step
        )

    # Diagnostic visualization
    visual_dir = os.path.join(experiment_dir, "visualization")
    os.makedirs(visual_dir, exist_ok=True)
    
    np.save(os.path.join(visual_dir, "vp.npy"), vp.data) 
    np.save(os.path.join(visual_dir, "transducer_locations.npy"), problem.geometry.coordinates)

    # ------------- Forward simulation -------------

    # Create the PDE (iso-acoustic wave equation)
    pde = IsoAcousticDevito.remote(grid=problem.grid)
    await forward(
        problem, pde, vp, 
        kernel="OT4", 
        dump_forward_wavefield=25,
        dump_wavefield_id=0 
    )

    # Save acquisitions
    acq_path = os.path.join(experiment_dir, f"{name}-Acquisitions.h5")
    problem.acquisitions.dump(filename=acq_path)
    logger.info("Acquisitions saved to %s", acq_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forward modeling for one brain.")
    parser.add_argument(
        "--brain", type=str, default=os.environ.get("MODEL_NAME", "vp_996782"),
            help="Brain model name (e.g. vp_996782)"
    )
    args, _ = parser.parse_known_args()
    brain = args.brain
    
    mosaic.run(main, log_level="perf")
