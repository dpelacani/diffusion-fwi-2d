from stride import *
from stride.utils import wavelets, fetch
import numpy as np


async def main(runtime):
    # Create the grid
    shape = (320, 256)
    extra = (50, 50)
    absorbing = (40, 40)
    spacing = (0.5e-3, 0.5e-3)

    space = Space(shape=shape, extra=extra, absorbing=absorbing, spacing=spacing)

    start = 0.0
    step = 0.08e-6
    num = 2500

    time = Time(start=start, step=step, num=num)

    # Create problem
    problem = Problem(name="anastasio2D", space=space, time=time)

    # Create medium
    # this is the speed of sound of the region of interest
    # vp contains a Numpy array with the velocity for every point on the grid
    vp = ScalarField(name="vp", grid=problem.grid)
    vp.load("BrainTrueModel.h5")

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
    # here we will create 32 receivers per shot
    num_receivers = 32
    step = num_locations // num_receivers
    for source in problem.geometry.locations:
        receivers = []
        for r in range(num_receivers):
            if (source.id + r * step) % num_locations != source.id:
                receivers.append(
                    problem.geometry.locations[(source.id + r * step) % num_locations]
                )

        shot = Shot(
            source.id,
            sources=[source],
            receivers=receivers,
            geometry=problem.geometry,
            problem=problem,
        )

        problem.acquisitions.add(shot)

    # Create wavelets
    # a tone burst is a common wavelet used in medical imaging
    f_centre = 0.25e6
    n_cycles = 3
    for shot in problem.acquisitions.shots:
        shot.wavelets.data[0, :] = wavelets.tone_burst(
            f_centre, n_cycles, time.num, time.step
        )

    # Create the PDE
    # the physics of our problem are represented by the iso-acoustic wave equation
    pde = IsoAcousticDevito.remote(grid=problem.grid)

    # Run
    # this will generate the simulated observed data and save it to a file
    await forward(problem, pde, vp, kernel="OT4", platform="nvidia-acc")
    for shot in problem.acquisitions.shots:
        data = shot.observed.data
        # Add noise to the observed data
        # 0.9 is the maximum amplitude of the wavelet, we set it as the standard deviation of the noise
        # to avoid the noise being too large, we only use 1% of the noise
        noise = 0.01 * np.random.normal(loc=0.0, scale=0.9, size=data.shape)
        shot.observed.data[:] = shot.observed.data + noise
        shot.append_observed(path=problem.output_folder, project_name=problem.name)


if __name__ == "__main__":
    mosaic.run(main)
