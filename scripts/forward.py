import os
from stride import *
from stride.utils import wavelets

from diffusionfwi.fwi.utils import npy2h5

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers.clear()  # Clear any existing handlers


async def main(runtime):
    # Create the spatio-temporal grid
    shape = (320, 256)
    extra = (50, 50)
    absorbing = (40, 40)
    spacing = (0.5e-3, 0.5e-3)

    space = Space(shape=shape, extra=extra, absorbing=absorbing, spacing=spacing)

    start = 0.0
    step = 0.08e-6
    num = 2500

    time = Time(start=start, step=step, num=num)

    true_model = "vp_996782.npy"
    name = true_model.split(".")[0].upper()
    experiment_dir = f"./exps/forward/{name}"

    # Create h5 from npy true model
    if not os.path.isfile(f"{experiment_dir}/{name}.h5"):
        npy2h5(
            path=f"/scratch_hive/dp4018/data/ultrasound-data/Ultrasound-Vp-axial-models/{true_model}",
            name=f"{name}.h5",
            save_path=experiment_dir,
            t_num=num,
            t_step=step,
            t_start=start,
            extra=extra,
            absorbing=absorbing,
            spacing=spacing,
        )  # After this a file called {name}.h5 will be created in the experiment directory
    logger.info(f"Created HDF5 file for true model at {experiment_dir}/{name}.h5")

    # Create problem -  this is the base of how Stride operates
    problem = Problem(
        name=name,
        input_folder=experiment_dir,
        output_folder=experiment_dir,
        space=space,
        time=time,
    )

    # Create medium
    # this is the speed of sound of the region of interest
    # vp contains a Numpy array with the velocity for every point on the grid
    vp = ScalarField(name="vp", grid=problem.grid)
    vp.load(f"{experiment_dir}/{name}.h5")

    problem.medium.add(vp)

    # Create transducers
    # we generally assume point transducers in simulation, but
    # in reality more complex transducer geometries are used
    problem.transducers.default()

    # Create geometry
    # this will make a ring of 256 transducers around the region of interest
    num_locations = 256
    problem.geometry.default(
        "elliptical",
        num_locations,
        radius=((space.limit[0] - 7.0e-3) / 2, (space.limit[1] - 5.0e-3) / 2),
    )

    # Create acquisitions
    # this will link every transducer in the region of interest with each other:
    # they will each fire a pulse in turns, and then record the data for every pulse
    # we call each of these turns a shot
    problem.acquisitions.default()

    # Create wavelets
    # a tone burst is a common wavelet used in medical imaging
    f_centre = 0.25e6
    n_cycles = 3

    for shot in problem.acquisitions.shots:
        shot.wavelets.data[0, :] = wavelets.tone_burst(
            f_centre, n_cycles, time.num, time.step
        )

    # Plot
    # some plotting of what we have built so far
    # problem.plot()

    # Create the PDE
    # the physics of our problem are represented by the iso-acoustic wave equation
    pde = IsoAcousticDevito.remote(grid=problem.grid)

    # Run
    # this will generate the simulated observed data and save it to a file
    await forward(problem, pde, vp, kernel="OT4", platform="nvidia-acc")

    # Optional - we can add noise to the observed data to make the problem more realistic
    # 0.9 is the maximum amplitude of the wavelet, we set it as the standard deviation of the noise
    # to avoid the noise being too large, we only use 1% of the noise
    # for shot in problem.acquisitions.shots:
    #     data = shot.observed.data
    #     noise = 0.01 * np.random.normal(loc=0.0, scale=0.9, size=data.shape)
    #     shot.observed.data[:] = shot.observed.data #+ noise
    #     shot.append_observed(path=problem.output_folder, project_name=problem.name)

    problem.acquisitions.dump(filename=f"{experiment_dir}/{name}-Acquisitions.h5")


if __name__ == "__main__":
    mosaic.run(main, log_level="perf")
