import os
import warnings

import numpy
from packaging import version

from relentless import mpi
from relentless.collections import FixedKeyDict, PairMatrix
from relentless.math import Interpolator
from relentless.model import ensemble, extent

from . import initialize, md, simulate

try:
    import hoomd
    import hoomd.md

    _hoomd_found = True
except ImportError:
    _hoomd_found = False

if _hoomd_found:
    try:
        _version = hoomd.version.version
    except AttributeError:
        _version = hoomd.__version__
    _version = version.parse(_version)

else:
    _version = None

if _hoomd_found and _version.major == 3:
    from hoomd.custom import Action
else:
    # we need to spoof this Action class to use later in v2 callbacks
    class Action:
        pass


try:
    import freud

    _freud_found = True
except ImportError:
    _freud_found = False


class SimulationOperation(simulate.SimulationOperation):
    def __call__(self, sim):
        if _version.major == 3:
            self._call_v3(sim)
        elif _version.major == 2:
            self._call_v2(sim)
        else:
            raise NotImplementedError(f"HOOMD {_version} not supported")

    def _call_v3(self, sim):
        raise NotImplementedError(
            f"{self.__class__.__name__} not implemented in HOOMD 3"
        )

    def _call_v2(self, sim):
        raise NotImplementedError(
            f"{self.__class__.__name__} not implemented in HOOMD 2"
        )


class AnalysisOperation(simulate.AnalysisOperation):
    def pre_run(self, sim, sim_op):
        if _version.major == 3:
            self._pre_run_v3(sim, sim_op)
        elif _version.major == 2:
            self._pre_run_v2(sim, sim_op)
        else:
            raise NotImplementedError(f"HOOMD {_version} not supported")

    def post_run(self, sim, sim_op):
        if _version.major == 3:
            self._post_run_v3(sim, sim_op)
        elif _version.major == 2:
            self._post_run_v2(sim, sim_op)
        else:
            raise NotImplementedError(f"HOOMD {_version} not supported")

    def _pre_run_v3(self, sim, sim_op):
        raise NotImplementedError(
            f"{self.__class__.__name__} not implemented in HOOMD 3"
        )

    def _pre_run_v2(self, sim, sim_op):
        raise NotImplementedError(
            f"{self.__class__.__name__} not implemented in HOOMD 2"
        )

    def _post_run_v3(self, sim, sim_op):
        raise NotImplementedError(
            f"{self.__class__.__name__} not implemented in HOOMD 3"
        )

    def _post_run_v2(self, sim, sim_op):
        raise NotImplementedError(
            f"{self.__class__.__name__} not implemented in HOOMD 2"
        )


# initializers
class InitializationOperation(simulate.InitializationOperation, SimulationOperation):
    def _call_v3(self, sim):
        self._initialize_v3(sim)
        sim.dimension = sim["engine"]["_hoomd"].state.box.dimensions
        sim.types = sim["engine"]["_hoomd"].state.particle_types

        # parse masses by type
        snap = sim["engine"]["_hoomd"].state.get_snapshot()
        sim.masses = self._get_masses_from_snapshot(sim, snap)

        # create the potentials, defer attaching until later
        neighbor_list = hoomd.md.nlist.Tree(buffer=sim.potentials.pair.neighbor_buffer)
        pair_potential = hoomd.md.pair.Table(nlist=neighbor_list)
        for i, j in sim.pairs:
            r = sim.potentials.pair.x
            u = sim.potentials.pair.energy((i, j))
            f = sim.potentials.pair.force((i, j))
            if numpy.any(numpy.isinf(u)) or numpy.any(numpy.isinf(f)):
                raise ValueError("Pair potential/force is infinite at evaluated r")
            pair_potential.params[(i, j)] = dict(r_min=r[0], U=u, F=f)
            # this could be trimmed shorter, as in lammps
            pair_potential.r_cut[(i, j)] = r[-1]
        sim[self]["_potentials"] = [pair_potential]

    def _call_v2(self, sim):
        # initialize
        sim[self]["_system"] = self._initialize_v2(sim)
        sim.dimension = sim[self]["_system"].box.dimensions
        sim.types = sim[self]["_system"].particles.types

        # parse masses by type

        with sim["engine"]["_hoomd"]:
            snap = sim[self]["_system"].take_snapshot(particles=True)
            sim.masses = self._get_masses_from_snapshot(sim, snap)

        # attach the potentials
        def _table_eval(r_i, rmin, rmax, **coeff):
            r = coeff["r"]
            u = coeff["u"]
            f = coeff["f"]
            u_r = Interpolator(r, u)
            f_r = Interpolator(r, f)
            return (u_r(r_i), f_r(r_i))

        with sim["engine"]["_hoomd"]:
            neighbor_list = hoomd.md.nlist.tree(
                r_buff=sim.potentials.pair.neighbor_buffer
            )
            pair_potential = hoomd.md.pair.table(
                width=len(sim.potentials.pair.x), nlist=neighbor_list
            )
            for i, j in sim.pairs:
                r = sim.potentials.pair.x
                u = sim.potentials.pair.energy((i, j))
                f = sim.potentials.pair.force((i, j))
                if numpy.any(numpy.isinf(u)) or numpy.any(numpy.isinf(f)):
                    raise ValueError("Pair potential/force is infinite at evaluated r")
                pair_potential.pair_coeff.set(
                    i,
                    j,
                    func=_table_eval,
                    rmin=r[0],
                    rmax=r[-1],
                    coeff=dict(r=r, u=u, f=f),
                )

    def _initialize_v3(self, sim):
        raise NotImplementedError(
            f"{self.__class__.__name__} not implemented in HOOMD 3"
        )

    def _initialize_v2(self, sim):
        raise NotImplementedError(
            f"{self.__class__.__name__} not implemented in HOOMD 2"
        )

    def _get_masses_from_snapshot(self, sim, snap):
        masses = FixedKeyDict(sim.types)
        masses_ = {}
        # snapshot is only valid on root, so read there and broadcast
        if mpi.world.rank == 0:
            for i in sim.types:
                mi = snap.particles.mass[
                    snap.particles.typeid == snap.particles.types.index(i)
                ]
                if len(mi) == 0:
                    raise KeyError(f"Type {i} not present in simulation")
                elif not numpy.all(mi == mi[0]):
                    raise ValueError("All masses for a type must be equal")
                masses_[i] = mi[0]
        masses_ = mpi.world.bcast(masses_, root=0)
        masses.update(masses_)
        return masses


class InitializeFromFile(InitializationOperation):
    """Initialize a simulation from a GSD file.

    Parameters
    ----------
    filename : str
        The file from which to read the system data.

    """

    def __init__(self, filename):
        self.filename = os.path.realpath(filename)

    def _initialize_v3(self, sim):
        sim["engine"]["_hoomd"].create_state_from_gsd(self.filename)

    def _initialize_v2(self, sim):
        with sim["engine"]["_hoomd"]:
            return hoomd.init.read_gsd(self.filename)


class InitializeRandomly(InitializationOperation):
    """Initialize a randomly generated simulation box.

    If ``diameters`` is ``None``, the particles are randomly placed in the box.
    This can work pretty well for low densities, particularly if
    :class:`MinimizeEnergy` is used to remove overlaps before starting to run a
    simulation. However, it will typically fail for higher densities, where
    there are many overlaps that are hard to resolve.

    If ``diameters`` is specified for each particle type, the particles will
    be randomly packed into sites of a close-packed lattice. The insertion
    order is from big to small. No particles are allowed to overlap based on
    the diameters, which typically means the initially state will be more
    favorable than using random initialization. However, the packing procedure
    can fail if there is not enough room in the box to fit particles using
    lattice sites.

    Parameters
    ----------
    seed : int
        The seed to randomly initialize the particle locations.
    N : dict
        Number of particles of each type.
    V : :class:`~relentless.extent.Extent`
        Simulation extent.
    T : float
        Temperature. None means system is not thermalized.
    masses : dict
        Masses of each particle type. None means particles
        have unit mass.
    diameters : dict
        Diameter of each particle type. None means particles
        are randomly inserted without checking their sizes.

    """

    def __init__(self, seed, N, V, T, masses, diameters):
        self.seed = seed
        self.N = N
        self.V = V
        self.T = T
        self.masses = masses
        self.diameters = diameters

    def _initialize_v3(self, sim):
        snap = self._make_snapshot(sim)
        sim["engine"]["_hoomd"].create_state_from_snapshot(snap)

    def _initialize_v2(self, sim):
        with sim["engine"]["_hoomd"]:
            snap = self._make_snapshot(sim)
            return hoomd.init.read_snapshot(snap)

    def _make_snapshot(self, sim):
        # make the box and snapshot
        if isinstance(self.V, extent.TriclinicBox):
            box_array = self.V.as_array("HOOMD")
            if _version.major == 3:
                box = hoomd.Box(*box_array)
            elif _version.major == 2:
                box = hoomd.data.boxdim(*box_array, dimensions=3)
            else:
                raise NotImplementedError(f"HOOMD {_version} not supported")
        elif isinstance(self.V, extent.ObliqueArea):
            Lx, Ly, xy = self.V.as_array("HOOMD")
            if _version.major == 3:
                box = hoomd.Box(Lx=Lx, Ly=Ly, xy=xy)
            elif _version.major == 2:
                box = hoomd.data.boxdim(
                    Lx=Lx, Ly=Ly, Lz=1, xy=xy, xz=0, yz=0, dimensions=2
                )
            else:
                raise NotImplementedError(f"HOOMD {_version} not supported")
        else:
            raise ValueError("HOOMD supports 2d and 3d simulations")

        types = tuple(self.N.keys())
        typeids = {i: typeid for typeid, i in enumerate(types)}
        if _version.major == 3:
            snap = hoomd.Snapshot(communicator=mpi.world.comm)
            snap.configuration.box = box
            snap.particles.N = sum(self.N.values())
            snap.particles.types = list(types)
        elif _version.major == 2:
            snap = hoomd.data.make_snapshot(
                N=sum(self.N.values()), box=box, particle_types=list(types)
            )
        else:
            raise NotImplementedError(f"HOOMD {_version} not supported")

        # randomly place particles in fractional coordinates
        if mpi.world.rank == 0:
            # generate the positions and types
            if self.diameters is not None:
                (positions, all_types,) = initialize.InitializeRandomly._pack_particles(
                    self.seed, self.N, self.V, self.diameters
                )
            else:
                (
                    positions,
                    all_types,
                ) = initialize.InitializeRandomly._random_particles(
                    self.seed, self.N, self.V
                )

            # set the positions
            snap.particles.position[:, : box.dimensions] = positions

            # set the typeids
            snap.particles.typeid[:] = [typeids[i] for i in all_types]

            # set masses, defaulting to unit mass
            if self.masses is not None:
                snap.particles.mass[:] = [self.masses[i] for i in all_types]
            else:
                snap.particles.mass[:] = 1.0

            # set velocities, defaulting to zeros
            vel = numpy.zeros((snap.particles.N, 3))
            if self.T is not None:
                rng = numpy.random.default_rng(self.seed + 1)
                # Maxwell-Boltzmann = normal with variance kT/m per component
                v_mb = rng.normal(
                    scale=numpy.sqrt(sim.potentials.kB * self.T),
                    size=(snap.particles.N, box.dimensions),
                )
                v_mb /= numpy.sqrt(snap.particles.mass[:, None])

                # zero the linear momentum
                p_mb = numpy.sum(snap.particles.mass[:, None] * v_mb, axis=0)
                v_cm = p_mb / numpy.sum(snap.particles.mass)
                v_mb -= v_cm

                vel[:, : box.dimensions] = v_mb
            snap.particles.velocity[:] = vel

        return snap


# integrators
class MinimizeEnergy(SimulationOperation):
    """Run energy minimization until convergence.

    Valid **options** include:

    - **max_displacement** (`float`) - the maximum time step size the minimizer
      is allowed to use.
    - **steps_per_iteration** (`int`) - the number of steps the minimizer runs
      per iteration. Defaults to 100.

    Parameters
    ----------
    energy_tolerance : float
        Energy convergence criterion.
    force_tolerance : float
        Force convergence criterion.
    max_iterations : int
        Maximum number of iterations to run the minimization.
    options : dict
        Additional options for energy minimizer.

    Raises
    ------
    KeyError
        If a value for the maximum displacement is not provided.
    RuntimeError
        If energy minimization has failed to converge within the maximum
        number of iterations.

    """

    def __init__(self, energy_tolerance, force_tolerance, max_iterations, options):
        self.energy_tolerance = energy_tolerance
        self.force_tolerance = force_tolerance
        self.max_iterations = max_iterations
        self.options = options
        if "max_displacement" not in self.options:
            raise KeyError("HOOMD energy minimizer requires max_displacement option.")
        if "steps_per_iteration" not in self.options:
            self.options["steps_per_iteration"] = 100

    def _call_v3(self, sim):
        # setup FIRE minimization
        with warnings.catch_warnings():
            if _version >= version.Version("3.8"):
                warnings.simplefilter(action="ignore", category=FutureWarning)
            nve = hoomd.md.methods.NVE(filter=hoomd.filter.All())
        fire = hoomd.md.minimize.FIRE(
            dt=self.options["max_displacement"],
            force_tol=self.force_tolerance,
            angmom_tol=self.force_tolerance,
            energy_tol=self.energy_tolerance,
            forces=sim[sim.initializer]["_potentials"],
            methods=[nve],
        )
        sim["engine"]["_hoomd"].operations.integrator = fire

        # run while not yet converged
        it = 0
        while not fire.converged and it < self.max_iterations:
            sim["engine"]["_hoomd"].run(self.options["steps_per_iteration"])
            it += 1
        if not fire.converged:
            raise RuntimeError("Energy minimization failed to converge.")

        # cleanup
        sim["engine"]["_hoomd"].operations.integrator = None
        del fire, nve

    def _call_v2(self, sim):
        with sim["engine"]["_hoomd"]:
            # setup FIRE minimization
            fire = hoomd.md.integrate.mode_minimize_fire(
                dt=self.options["max_displacement"],
                Etol=self.energy_tolerance,
                ftol=self.force_tolerance,
            )
            all_ = hoomd.group.all()
            nve = hoomd.md.integrate.nve(all_)

            # run while not yet converged
            it = 0
            while not fire.has_converged() and it < self.max_iterations:
                hoomd.run(self.options["steps_per_iteration"])
                it += 1
            if not fire.has_converged():
                raise RuntimeError("Energy minimization failed to converge.")

            # try to cleanup these integrators from the system
            # we want them to ideally be isolated to this method
            nve.disable()
            del nve
            del fire


class _MDIntegrator(SimulationOperation):
    """Base HOOMD molecular dynamics integrator.

    Parameters
    ----------
    steps : int
        Number of simulation time steps.
    timestep : float
        Simulation time step.
    analyzers : :class:`~relentless.simulate.AnalysisOperation` or list
        Analysis operations to perform with run (defaults to ``None``).

    """

    def __init__(self, steps, timestep, analyzers):
        self.steps = steps
        self.timestep = timestep
        self.analyzers = analyzers

    @property
    def analyzers(self):
        return self._analyzers

    @analyzers.setter
    def analyzers(self, ops):
        if ops is not None:
            try:
                ops_ = list(ops)
            except TypeError:
                ops_ = [ops]
        else:
            ops_ = []

        self._analyzers = ops_

    @staticmethod
    def _make_kT(sim, thermostat, steps):
        """Cast thermostat into a kT parameter for HOOMD integrators."""
        # force the type of thermostat, in case it's a float
        if not isinstance(thermostat, md.Thermostat):
            thermostat = md.Thermostat(thermostat)

        kB = sim.potentials.kB
        if thermostat.anneal:
            if steps > 1:
                if _version.major == 3:
                    return hoomd.variant.Ramp(
                        A=kB * thermostat.T[0],
                        B=kB * thermostat.T[1],
                        t_start=sim["engine"]["_hoomd"].timestep,
                        t_ramp=steps - 1,
                    )
                elif _version.major == 2:
                    return hoomd.variant.linear_interp(
                        ((0, kB * thermostat.T[0]), (steps - 1, kB * thermostat.T[1])),
                        zero="now",
                    )
                else:
                    raise NotImplementedError(f"HOOMD {_version} not supported")
            else:
                # if only 1 step, use the end point value. this is a corner case
                return kB * thermostat.T[1]
        else:
            return kB * thermostat.T


class RunBrownianDynamics(_MDIntegrator):
    """Perform a Brownian dynamics simulation.

    See :class:`relentless.simulate.RunBrownianDynamics` for details.

    Parameters
    ----------
    steps : int
        Number of simulation time steps.
    timestep : float
        Simulation time step.
    T : float
        Temperature.
    friction : float
        Sets drag coefficient for each particle type.
    seed : int
        Seed used to randomly generate a uniform force.
    analyzers : :class:`~relentless.simulate.AnalysisOperation` or list
        Analysis operations to perform with run (defaults to ``None``).

    """

    def __init__(self, steps, timestep, T, friction, seed, analyzers):
        super().__init__(steps, timestep, analyzers)
        self.T = T
        self.friction = friction
        self.seed = seed

    def _call_v3(self, sim):
        kT = self._make_kT(sim, self.T, self.steps)
        bd = hoomd.md.methods.Brownian(filter=hoomd.filter.All(), kT=kT)
        for t in sim.types:
            try:
                gamma = self.friction[t]
            except TypeError:
                gamma = self.friction
            bd.gamma[t] = gamma
        ig = hoomd.md.Integrator(
            dt=self.timestep,
            forces=sim[sim.initializer]["_potentials"],
            methods=[bd],
        )
        sim["engine"]["_hoomd"].operations.integrator = ig
        sim["engine"]["_hoomd"].seed = self.seed

        # run + analysis
        for analyzer in self.analyzers:
            analyzer.pre_run(sim, self)

        sim["engine"]["_hoomd"].run(self.steps, write_at_start=True)

        for analyzer in self.analyzers:
            analyzer.post_run(sim, self)

        sim["engine"]["_hoomd"].operations.integrator = None
        del ig, bd

    def _call_v2(self, sim):
        with sim["engine"]["_hoomd"]:
            ig = hoomd.md.integrate.mode_standard(self.timestep)
            kT = self._make_kT(sim, self.T, self.steps)
            bd = hoomd.md.integrate.brownian(
                group=hoomd.group.all(), kT=kT, seed=self.seed
            )
            for t in sim.types:
                try:
                    gamma = self.friction[t]
                except TypeError:
                    gamma = self.friction
                bd.set_gamma(t, gamma)

            # run + analysis
            for analyzer in self.analyzers:
                analyzer.pre_run(sim, self)

            hoomd.run(self.steps)

            for analyzer in self.analyzers:
                analyzer.post_run(sim, self)

            bd.disable()
            del bd, ig


class RunLangevinDynamics(_MDIntegrator):
    """Perform a Langevin dynamics simulation.

    See :class:`relentless.simulate.RunLangevinDynamics` for details.

    Parameters
    ----------
    steps : int
        Number of simulation time steps.
    timestep : float
        Simulation time step.
    T : float
        Temperature.
    friction : float or dict
        Sets drag coefficient for each particle type (shared or per-type).
    seed : int
        Seed used to randomly generate a uniform force.
    analyzers : :class:`~relentless.simulate.AnalysisOperation` or list
        Analysis operations to perform with run (defaults to ``None``).

    """

    def __init__(self, steps, timestep, T, friction, seed, analyzers):
        super().__init__(steps, timestep, analyzers)
        self.T = T
        self.friction = friction
        self.seed = seed

    def _call_v3(self, sim):
        kT = self._make_kT(sim, self.T, self.steps)
        ld = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=kT)
        for t in sim.types:
            try:
                gamma = self.friction[t]
            except TypeError:
                gamma = self.friction
            ld.gamma[t] = gamma
        ig = hoomd.md.Integrator(
            dt=self.timestep,
            forces=sim[sim.initializer]["_potentials"],
            methods=[ld],
        )
        sim["engine"]["_hoomd"].operations.integrator = ig
        sim["engine"]["_hoomd"].seed = self.seed

        # run + analysis
        for analyzer in self.analyzers:
            analyzer.pre_run(sim, self)

        sim["engine"]["_hoomd"].run(self.steps, write_at_start=True)

        for analyzer in self.analyzers:
            analyzer.post_run(sim, self)

        sim["engine"]["_hoomd"].operations.integrator = None
        del ig, ld

    def _call_v2(self, sim):
        with sim["engine"]["_hoomd"]:
            ig = hoomd.md.integrate.mode_standard(self.timestep)
            kT = self._make_kT(sim, self.T, self.steps)
            ld = hoomd.md.integrate.langevin(
                group=hoomd.group.all(), kT=kT, seed=self.seed
            )
            for t in sim.types:
                try:
                    gamma = self.friction[t]
                except TypeError:
                    gamma = self.friction
                ld.set_gamma(t, gamma)

            # run + analysis
            for analyzer in self.analyzers:
                analyzer.pre_run(sim, self)

            hoomd.run(self.steps)

            for analyzer in self.analyzers:
                analyzer.post_run(sim, self)

            ld.disable()
            del ld, ig


class RunMolecularDynamics(_MDIntegrator):
    """Perform a molecular dynamics simulation.

    This method supports:

    - NVE integration
    - NVT integration with Nosé-Hoover or Berendsen thermostat
    - NPH integration with MTK barostat
    - NPT integration with Nosé-Hoover thermostat and MTK barostat

    Parameters
    ----------
    steps : int
        Number of simulation time steps.
    timestep : float
        Simulation time step.
    thermostat : :class:`~relentless.simulate.Thermostat`
        Thermostat for temperature control. None means no thermostat.
    barostat : :class:`~relentless.simulate.Barostat`
        Barostat for pressure control. None means no barostat.
    analyzers : :class:`~relentless.simulate.AnalysisOperation` or list
        Analysis operations to perform with run (defaults to ``None``).

    Raises
    ------
    TypeError
        If an appropriate combination of thermostat and barostat is not set.

    """

    def __init__(self, steps, timestep, thermostat, barostat, analyzers):
        super().__init__(steps, timestep, analyzers)
        self.thermostat = thermostat
        self.barostat = barostat

    def _call_v3(self, sim):
        if self.thermostat is not None:
            kT = self._make_kT(sim, self.thermostat, self.steps)
        else:
            kT = None

        if self.thermostat is None and self.barostat is None:
            with warnings.catch_warnings():
                if _version >= version.Version("3.8"):
                    warnings.simplefilter(action="ignore", category=FutureWarning)
                ig_method = hoomd.md.methods.NVE(filter=hoomd.filter.All())
        elif (
            isinstance(self.thermostat, md.BerendsenThermostat)
            and self.barostat is None
        ):
            with warnings.catch_warnings():
                if _version >= version.Version("3.8"):
                    warnings.simplefilter(action="ignore", category=FutureWarning)
                ig_method = hoomd.md.methods.Berendsen(
                    filter=hoomd.filter.All(),
                    kT=kT,
                    tau=self.thermostat.tau,
                )
        elif (
            isinstance(self.thermostat, md.NoseHooverThermostat)
            and self.barostat is None
        ):
            with warnings.catch_warnings():
                if _version >= version.Version("3.8"):
                    warnings.simplefilter(action="ignore", category=FutureWarning)
                ig_method = hoomd.md.methods.NVT(
                    filter=hoomd.filter.All(),
                    kT=kT,
                    tau=self.thermostat.tau,
                )
        elif self.thermostat is None and isinstance(self.barostat, md.MTKBarostat):
            with warnings.catch_warnings():
                if _version >= version.Version("3.8"):
                    warnings.simplefilter(action="ignore", category=FutureWarning)
                ig_method = hoomd.md.methods.NPH(
                    filter=hoomd.filter.All(),
                    S=self.barostat.P,
                    tauS=self.barostat.tau,
                    couple="xyz" if sim.dimension == 3 else "xy",
                )
        elif isinstance(self.thermostat, md.NoseHooverThermostat) and isinstance(
            self.barostat, md.MTKBarostat
        ):
            with warnings.catch_warnings():
                if _version >= version.Version("3.8"):
                    warnings.simplefilter(action="ignore", category=FutureWarning)
                ig_method = hoomd.md.methods.NPT(
                    filter=hoomd.filter.All(),
                    kT=kT,
                    tau=self.thermostat.tau,
                    S=self.barostat.P,
                    tauS=self.barostat.tau,
                    couple="xyz" if sim.dimension == 3 else "xy",
                )
        else:
            raise TypeError(
                "An appropriate combination of thermostat and barostat must be set."
            )

        ig = hoomd.md.Integrator(
            dt=self.timestep,
            forces=sim[sim.initializer]["_potentials"],
            methods=[ig_method],
        )
        sim["engine"]["_hoomd"].operations.integrator = ig

        # run + analysis
        for analyzer in self.analyzers:
            analyzer.pre_run(sim, self)

        sim["engine"]["_hoomd"].run(self.steps, write_at_start=True)

        for analyzer in self.analyzers:
            analyzer.post_run(sim, self)

        sim["engine"]["_hoomd"].operations.integrator = None
        del ig, ig_method

    def _call_v2(self, sim):
        with sim["engine"]["_hoomd"]:
            ig = hoomd.md.integrate.mode_standard(self.timestep)
            if self.thermostat is not None:
                kT = self._make_kT(sim, self.thermostat, self.steps)
            else:
                kT = None
            if self.thermostat is None and self.barostat is None:
                ig_method = hoomd.md.integrate.nve(group=hoomd.group.all())
            elif (
                isinstance(self.thermostat, md.BerendsenThermostat)
                and self.barostat is None
            ):
                ig_method = hoomd.md.integrate.berendsen(
                    group=hoomd.group.all(),
                    kT=kT,
                    tau=self.thermostat.tau,
                )
            elif (
                isinstance(self.thermostat, md.NoseHooverThermostat)
                and self.barostat is None
            ):
                ig_method = hoomd.md.integrate.nvt(
                    group=hoomd.group.all(),
                    kT=kT,
                    tau=self.thermostat.tau,
                )
            elif self.thermostat is None and isinstance(self.barostat, md.MTKBarostat):
                ig_method = hoomd.md.integrate.nph(
                    group=hoomd.group.all(), P=self.barostat.P, tauP=self.barostat.tau
                )
            elif isinstance(self.thermostat, md.NoseHooverThermostat) and isinstance(
                self.barostat, md.MTKBarostat
            ):
                ig_method = hoomd.md.integrate.npt(
                    group=hoomd.group.all(),
                    kT=kT,
                    tau=self.thermostat.tau,
                    P=self.barostat.P,
                    tauP=self.barostat.tau,
                )
            else:
                raise TypeError(
                    "An appropriate combination of thermostat and barostat must be set."
                )

            # run + analysis
            for analyzer in self.analyzers:
                analyzer.pre_run(sim, self)

            hoomd.run(self.steps)

            for analyzer in self.analyzers:
                analyzer.post_run(sim, self)

            ig_method.disable()
            del ig_method, ig


# analyzers
class EnsembleAverage(AnalysisOperation):
    """Analyzer for the simulation ensemble.

    The simulation ensemble is analyzed online while it is running. The
    instantaneous thermodynamic properties are extracted from a HOOMD logger,
    while the radial distribution function is computed using :mod:`freud`.

    Parameters
    ----------
    check_thermo_every : int
        Number of timesteps between computing thermodynamic properties.
    check_rdf_every : int
        Number of time steps between computing the RDF.
    rdf_dr : float
        The width of a bin in the RDF histogram.

    """

    def __init__(self, check_thermo_every, check_rdf_every, rdf_dr):
        self.check_thermo_every = check_thermo_every
        self.check_rdf_every = check_rdf_every
        self.rdf_dr = rdf_dr

    def _pre_run_v3(self, sim, sim_op):
        # thermodynamic properties
        sim[self]["_thermo"] = hoomd.md.compute.ThermodynamicQuantities(
            hoomd.filter.All()
        )
        sim[self]["_thermo_callback"] = self.ThermodynamicsCallback(
            thermo=sim[self]["_thermo"],
            dimension=sim.dimension,
        )
        sim[self]["_hoomd_thermo_callback"] = hoomd.write.CustomWriter(
            trigger=self.check_thermo_every,
            action=sim[self]["_thermo_callback"],
        )
        sim["engine"]["_hoomd"].operations.computes.append(sim[self]["_thermo"])
        sim["engine"]["_hoomd"].operations.writers.append(
            sim[self]["_hoomd_thermo_callback"]
        )

        # RDF
        rdf_params = self._get_rdf_params(sim)
        sim[self]["_rdf_callback"] = self.RDFCallback(
            system=sim["engine"]["_hoomd"].state,
            params=rdf_params,
        )
        sim[self]["_hoomd_rdf_callback"] = hoomd.write.CustomWriter(
            trigger=self.check_rdf_every,
            action=sim[self]["_rdf_callback"],
        )
        sim["engine"]["_hoomd"].operations.writers.append(
            sim[self]["_hoomd_rdf_callback"]
        )

    def _pre_run_v2(self, sim, sim_op):
        with sim["engine"]["_hoomd"]:
            # thermodynamic properties
            if sim.dimension == 3:
                quantities_logged = [
                    "temperature",
                    "pressure",
                    "lx",
                    "ly",
                    "lz",
                    "xy",
                    "xz",
                    "yz",
                ]
            elif sim.dimension == 2:
                quantities_logged = ["temperature", "pressure", "lx", "ly", "xy"]
            else:
                raise ValueError("HOOMD only supports 2d or 3d simulations")
            sim[self]["_thermo"] = hoomd.analyze.log(
                filename=None,
                quantities=quantities_logged,
                period=self.check_thermo_every,
            )
            sim[self]["_thermo_callback"] = self.ThermodynamicsCallback(
                thermo=sim[self]["_thermo"],
                dimension=sim.dimension,
            )
            sim[self]["_hoomd_thermo_callback"] = hoomd.analyze.callback(
                callback=sim[self]["_thermo_callback"].act,
                period=self.check_thermo_every,
            )

            # pair distribution function
            rdf_params = self._get_rdf_params(sim)
            sim[self]["_rdf_callback"] = self.RDFCallback(
                system=sim[sim.initializer]["_system"],
                params=rdf_params,
            )
            sim[self]["_hoomd_rdf_callback"] = hoomd.analyze.callback(
                callback=sim[self]["_rdf_callback"].act, period=self.check_rdf_every
            )

    def _post_run_v3(self, sim, sim_op):
        sim["engine"]["_hoomd"].operations.writers.remove(
            sim[self]["_hoomd_thermo_callback"]
        )
        sim["engine"]["_hoomd"].operations.computes.remove(sim[self]["_thermo"])
        del sim[self]["_hoomd_thermo_callback"], sim[self]["_thermo"]

        sim["engine"]["_hoomd"].operations.writers.remove(
            sim[self]["_hoomd_rdf_callback"]
        )
        del sim[self]["_hoomd_rdf_callback"]

    def _post_run_v2(self, sim, sim_op):
        with sim["engine"]["_hoomd"]:
            sim[self]["_hoomd_thermo_callback"].disable()
            sim[self]["_thermo"].disable()

            sim[self]["_hoomd_rdf_callback"].disable()

        del sim[self]["_hoomd_thermo_callback"], sim[self]["_thermo"]
        del sim[self]["_hoomd_rdf_callback"]

    def process(self, sim, sim_op):
        rdf_recorder = sim[self]["_rdf_callback"]
        thermo_recorder = sim[self]["_thermo_callback"]

        # make sure samples were collected
        if thermo_recorder.num_samples == 0:
            raise RuntimeError("No thermo samples were collected")
        if rdf_recorder.num_samples == 0:
            raise RuntimeError("No RDF samples were collected")

        ens = ensemble.Ensemble(
            T=thermo_recorder.T,
            N=rdf_recorder.N,
            V=thermo_recorder.V,
            P=thermo_recorder.P,
        )
        for pair in rdf_recorder.rdf:
            ens.rdf[pair] = rdf_recorder.rdf[pair]

        sim[self]["ensemble"] = ens
        sim[self]["num_thermo_samples"] = thermo_recorder.num_samples
        sim[self]["num_rdf_samples"] = rdf_recorder.num_samples

        # disable and delete callbacks
        del sim[self]["_thermo_callback"]
        del sim[self]["_rdf_callback"]

    def _get_rdf_params(self, sim):
        rdf_params = PairMatrix(sim.types)
        rmax = sim.potentials.pair.x[-1]
        bins = numpy.round(rmax / self.rdf_dr).astype(int)
        for pair in rdf_params:
            rdf_params[pair] = {"bins": bins, "rmax": rmax}
        return rdf_params

    class ThermodynamicsCallback(Action):
        """HOOMD callback for averaging thermodynamic properties."""

        if _version is not None and _version.major == 3:
            flags = [Action.Flags.PRESSURE_TENSOR]

        def __init__(self, thermo, dimension):
            if dimension not in (2, 3):
                raise ValueError("Only 2 or 3 dimensions supported")

            self.thermo = thermo
            self.dimension = dimension
            self.reset()

        def act(self, timestep):
            if _version.major == 3:
                self._T += self.thermo.kinetic_temperature
                self._P += self.thermo.pressure
                for key in self._V:
                    self._V[key] += getattr(self._state.box, key)
                self.num_samples += 1
            elif _version.major == 2:
                self._T += self.thermo.query("temperature")
                self._P += self.thermo.query("pressure")
                for key in self._V:
                    self._V[key] += self.thermo.query(key.lower())
                self.num_samples += 1
            else:
                raise NotImplementedError(f"HOOMD {_version} not supported")

        def reset(self):
            """Resets sample number, ``T``, ``P``, and all ``V`` parameters to 0."""
            self.num_samples = 0
            self._T = 0.0
            self._P = 0.0
            if self.dimension == 3:
                self._V = {
                    "Lx": 0.0,
                    "Ly": 0.0,
                    "Lz": 0.0,
                    "xy": 0.0,
                    "xz": 0.0,
                    "yz": 0.0,
                }
            elif self.dimension == 2:
                self._V = {"Lx": 0.0, "Ly": 0.0, "xy": 0.0}

        @property
        def T(self):
            """float: Average temperature across samples."""
            if self.num_samples > 0:
                return self._T / self.num_samples
            else:
                return None

        @property
        def P(self):
            """float: Average pressure across samples."""
            if self.num_samples > 0:
                return self._P / self.num_samples
            else:
                return None

        @property
        def V(self):
            """float: Average extent across samples."""
            if self.num_samples > 0:
                _V = {key: self._V[key] / self.num_samples for key in self._V}
                if self.dimension == 3:
                    return extent.TriclinicBox(**_V, convention="HOOMD")
                elif self.dimension == 2:
                    return extent.ObliqueArea(**_V, convention="HOOMD")
            else:
                return None

    class RDFCallback(Action):
        """HOOMD callback for averaging radial distribution function."""

        def __init__(self, system, params):
            self.system = system

            self.num_samples = 0
            self._N = FixedKeyDict(params.types)
            for i in self._N:
                self._N[i] = 0

            self._rdf = PairMatrix(params.types)
            for i, j in self._rdf:
                self._rdf[i, j] = freud.density.RDF(
                    bins=params[i, j]["bins"],
                    r_max=params[i, j]["rmax"],
                    normalize=(i == j),
                )

        def act(self, timestep):
            if _version.major == 3:
                snap = self.system.get_snapshot()
            elif _version.major == 2:
                snap = self.system.take_snapshot()
            else:
                raise NotImplementedError(f"HOOMD {_version} not supported")

            if mpi.world.rank == 0:
                if _version.major == 3:
                    dimensions = snap.configuration.dimensions
                    box_array = numpy.array(snap.configuration.box)
                elif _version.major == 2:
                    dimensions = snap.box.dimensions
                    box_array = numpy.array(
                        [
                            snap.box.Lx,
                            snap.box.Ly,
                            snap.box.Lz,
                            snap.box.xy,
                            snap.box.xz,
                            snap.box.yz,
                        ]
                    )
                    if snap.box.dimensions == 2:
                        box_array[2] = 0.0
                        box_array[-2:] = 0.0
                else:
                    raise NotImplementedError(f"HOOMD {_version} not supported")

                box = freud.box.Box.from_box(box_array, dimensions=dimensions)
                # pre build aabbs per type and count particles by type
                aabbs = {}
                type_masks = {}
                for i in self._N:
                    type_masks[i] = snap.particles.typeid == snap.particles.types.index(
                        i
                    )
                    self._N[i] += numpy.sum(type_masks[i])
                    aabbs[i] = freud.locality.AABBQuery(
                        box, snap.particles.position[type_masks[i]]
                    )
                # then do rdfs using the AABBs
                for i, j in self._rdf:
                    query_args = dict(self._rdf[i, j].default_query_args)
                    query_args.update(exclude_ii=(i == j))
                    # resetting when the samples are zero clears the RDF, on delay
                    self._rdf[i, j].compute(
                        aabbs[j],
                        snap.particles.position[type_masks[i]],
                        neighbors=query_args,
                        reset=(self.num_samples == 0),
                    )
            self.num_samples += 1

        @property
        def N(self):
            """:class:`~relentless.collections.FixedKeyDict`:
            Number of particles by type.
            """
            if self.num_samples > 0:
                N = FixedKeyDict(self._N.keys())
                for i in self._N:
                    if mpi.world.rank == 0:
                        Ni = self._N[i] / self.num_samples
                    else:
                        Ni = None
                    Ni = mpi.world.bcast(Ni, root=0)
                    N[i] = Ni
                return N
            else:
                return None

        @property
        def rdf(self):
            """:class:`~relentless.collections.PairMatrix`:
            Radial distribution functions.
            """
            if self.num_samples > 0:
                rdf = PairMatrix(self._rdf.types)
                for pair in rdf:
                    if mpi.world.rank == 0:
                        gr = numpy.column_stack(
                            (self._rdf[pair].bin_centers, self._rdf[pair].rdf)
                        )
                    else:
                        gr = None
                    gr = mpi.world.bcast_numpy(gr, root=0)
                    rdf[pair] = ensemble.RDF(gr[:, 0], gr[:, 1])
                return rdf
            else:
                return None

        def reset(self):
            """Reset the averages."""
            self.num_samples = 0
            for i in self._N.types:
                self._N[i] = 0
            # rdf will be reset on next call


class Record(AnalysisOperation):
    def __init__(self, quantities, every):
        self.quantities = quantities
        self.every = every

    def _pre_run_v3(self, sim, sim_op):
        sim[self]["_thermo"] = hoomd.md.compute.ThermodynamicQuantities(
            hoomd.filter.All()
        )
        sim[self]["_recorder"] = self.RecorderCallback(
            thermo=sim[self]["_thermo"],
            quantities=self.quantities,
        )
        sim[self]["_hoomd_recorder"] = hoomd.write.CustomWriter(
            trigger=self.every,
            action=sim[self]["_recorder"],
        )
        sim["engine"]["_hoomd"].operations.computes.append(sim[self]["_thermo"])
        sim["engine"]["_hoomd"].operations.writers.append(sim[self]["_hoomd_recorder"])

    def _pre_run_v2(self, sim, sim_op):
        with sim["engine"]["_hoomd"]:
            sim[self]["_thermo"] = hoomd.analyze.log(
                filename=None,
                quantities=self.quantities,
                period=self.every,
            )
            sim[self]["_recorder"] = self.RecorderCallback(
                thermo=sim[self]["_thermo"],
                quantities=self.quantities,
            )
            sim[self]["_hoomd_recorder"] = hoomd.analyze.callback(
                callback=sim[self]["_recorder"].act, period=self.every
            )

    def _post_run_v3(self, sim, sim_op):
        sim["engine"]["_hoomd"].operations.writers.remove(sim[self]["_hoomd_recorder"])
        sim["engine"]["_hoomd"].operations.computes.remove(sim[self]["_thermo"])
        del sim[self]["_hoomd_recorder"], sim[self]["_thermo"]

    def _post_run_v2(self, sim, sim_op):
        with sim["engine"]["_hoomd"]:
            sim[self]["_hoomd_recorder"].disable()
            sim[self]["_thermo"].disable()
        del sim[self]["_hoomd_recorder"], sim[self]["_thermo"]

    def process(self, sim, sim_op):
        recorder = sim[self]["_recorder"]
        sim[self]["timestep"] = recorder.timestep
        for q in self.quantities:
            values = numpy.array(getattr(recorder, q))
            # HOOMD actually reports kT as temperature, so scale by kB
            if q == "temperature":
                values /= sim.potentials.kB
            sim[self][q] = values

        del sim[self]["_recorder"]

    class RecorderCallback(Action):
        """HOOMD callback for recording thermodynamic properties."""

        if _version is not None and _version.major == 3:
            flags = [Action.Flags.PRESSURE_TENSOR]

        def __init__(self, thermo, quantities):
            self.thermo = thermo
            self.quantities = quantities
            self.reset()

        def act(self, timestep):
            self.timestep.append(timestep)
            for q in self.quantities:
                if _version.major == 3:
                    if q == "temperature":
                        val = getattr(self.thermo, "kinetic_temperature")
                    else:
                        val = getattr(self.thermo, q)
                elif _version.major == 2:
                    val = self.thermo.query(q)
                else:
                    raise NotImplementedError(f"HOOMD {_version} not supported")
                self._data[q].append(val)

        def __getattr__(self, item):
            if item in self.quantities:
                return self._data[item]
            else:
                return super().__getattr__(item)

        def reset(self):
            self.timestep = []
            self._data = {}
            for q in self.quantities:
                self._data[q] = []


class WriteTrajectory(AnalysisOperation):
    def __init__(self, filename, every, velocities, images, types, masses):
        self.filename = filename
        self.every = every
        self.velocities = velocities
        self.images = images
        self.types = types
        self.masses = masses

    def _pre_run_v3(self, sim, sim_op):
        sim[self]["_gsd"] = hoomd.write.GSD(
            trigger=self.every,
            filename=sim.directory.file(self.filename),
            filter=hoomd.filter.All(),
            mode="wb",
            dynamic=self._get_dynamic(),
        )
        sim["engine"]["_hoomd"].operations.writers.append(sim[self]["_gsd"])

    def _pre_run_v2(self, sim, sim_op):
        with sim["engine"]["_hoomd"]:
            # dump the .gsd file into the directory
            # Note: the .gsd file is overwritten for each call of the function
            sim[self]["_gsd"] = hoomd.dump.gsd(
                filename=sim.directory.file(self.filename),
                period=self.every,
                group=hoomd.group.all(),
                overwrite=True,
                dynamic=self._get_dynamic(),
            )

    def _post_run_v3(self, sim, sim_op):
        sim["engine"]["_hoomd"].operations.writers.remove(sim[self]["_gsd"])
        del sim[self]["_gsd"]

    def _post_run_v2(self, sim, sim_op):
        with sim["engine"]["_hoomd"]:
            sim[self]["_gsd"].disable()
        del sim[self]["_gsd"]

    def process(self, sim, sim_op):
        pass

    def _get_dynamic(self):
        # property group is always dyanmic in the trajectory file since it logs position
        dynamic = ["property"]
        # momentum group makes particle velocities and particles images dynamic
        if self.velocities is True or self.images is True:
            dynamic.append("momentum")
        return dynamic


class HOOMD(simulate.Simulation):
    """Simulation using HOOMD-blue.

    A simulation is performed using
    `HOOMD-blue <https://hoomd-blue.readthedocs.io/en/stable>`_.
    HOOMD-blue is a molecular dynamics program that can execute on both CPUs and
    GPUs, as a single process or with MPI parallelism. The launch configuration
    will be automatically selected for you when the simulation is run.

    Currently, this class only supports operations for HOOMD 2.x. Support for
    HOOMD 3.x will be added in future. The `freud <https://freud.readthedocs.io>`_
    analysis package (version 2.x) is also required for initialization and analysis.
    To use this simulation backend, you will need to install both :mod:`hoomd` and
    :mod:`freud` into your Python environment. :mod:`hoomd` is available through
    conda-forge or can be built from source, while :mod`freud` is available through
    both PyPI and conda-forge. Please refer to the package documentation for
    details of how to install these.

    .. warning::

        HOOMD requires that tabulated pair potentials be finite. A common place to
        have an infinite value is at :math:`r=0`, since potentials like
        :class:`~relentless.potential.pair.LennardJones` diverge there. You should
        make sure to exclude these values from the tabulated potentials,
        e.g., setting :attr:`~relentless.simulate.PairPotentialTabulator.rmin` to a
        small value larger than 0.

    Raises
    ------
    ImportError
        If the :mod:`hoomd` package is not found or is not version 2.x.
    ImportError
        If the :mod:`freud` package is not found or is not version 2.x.

    """

    def __init__(self, initializer, operations=None):
        if not _hoomd_found:
            raise ImportError("HOOMD not found.")

        if not _freud_found:
            raise ImportError("freud not found.")
        elif version.parse(freud.__version__).major != 2:
            raise ImportError("Only freud 2.x is supported.")

        super().__init__(initializer, operations)

    def _initialize_engine(self, sim):
        sim["engine"]["version"] = _version

        if _version.major == 3:
            device = hoomd.device.auto_select(
                communicator=mpi.world.comm, notice_level=0
            )
            sim["engine"]["_hoomd"] = hoomd.Simulation(device)
        elif _version.major == 2:
            # initialize hoomd exec conf once, then make a context
            if hoomd.context.exec_conf is None:
                hoomd.context.initialize("--notice-level=0")
                hoomd.util.quiet_status()
            sim["engine"]["_hoomd"] = hoomd.context.SimulationContext()
        else:
            raise NotImplementedError(f"HOOMD {_version} not supported")

    # initialize
    _InitializeFromFile = InitializeFromFile
    _InitializeRandomly = InitializeRandomly

    # md
    _MinimizeEnergy = MinimizeEnergy
    _RunBrownianDynamics = RunBrownianDynamics
    _RunLangevinDynamics = RunLangevinDynamics
    _RunMolecularDynamics = RunMolecularDynamics

    # analyze
    _EnsembleAverage = EnsembleAverage
    _Record = Record
    _WriteTrajectory = WriteTrajectory
