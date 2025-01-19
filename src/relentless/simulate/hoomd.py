import enum
import os
import shutil
import warnings

import freud
import gsd.hoomd
import lammpsio
import numpy
import packaging.version

from relentless import collections, mpi
from relentless.model import ensemble, extent

from . import analyze, initialize, md, simulate

try:
    import hoomd
    import hoomd.md

    _hoomd_found = True
except ImportError:
    _hoomd_found = False

if _hoomd_found:
    try:
        _hoomd_version = hoomd.version.version
    except AttributeError:
        _hoomd_version = hoomd.__version__
    _hoomd_version = packaging.version.parse(_hoomd_version)

    from hoomd.custom import Action

else:
    _hoomd_version = None

    class Action:
        class Flags(enum.IntEnum):
            PRESSURE_TENSOR = 0


try:
    _gsd_version = gsd.version.version
except AttributeError:
    _gsd_version = gsd.__version__
if packaging.version.Version(_gsd_version) >= packaging.version.Version("2.8.0"):
    _gsd_write_mode = "w"
else:
    _gsd_write_mode = "wb"


# initializers
class InitializationOperation(simulate.InitializationOperation):
    def __call__(self, sim):
        self._initialize(sim)
        sim.dimension = sim["engine"]["_hoomd"].state.box.dimensions
        sim.types = sim["engine"]["_hoomd"].state.particle_types

        # parse masses by type
        snap = sim["engine"]["_hoomd"].state.get_snapshot()
        sim.masses = self._get_masses_from_snapshot(sim, snap)
        self._assert_dimension_safe(sim, snap)
        # create the potentials, defer attaching until later
        exclusion = sim.potentials.pair.exclusions
        if exclusion is None:
            exclusion = ()
        elif "1-2" in exclusion:
            exclusion = ["bond" if ex == "1-2" else ex for ex in exclusion]
        neighbor_list = hoomd.md.nlist.Tree(
            buffer=sim.potentials.pair.neighbor_buffer,
            exclusions=exclusion,
        )
        pair_potential = hoomd.md.pair.Table(nlist=neighbor_list)
        r, u, f = sim.potentials.pair.pairwise_energy_and_force(
            sim.types, tight=True, minimum_num=2
        )
        for i, j in sim.pairs:
            if numpy.any(numpy.isinf(u[i, j])) or numpy.any(numpy.isinf(f[i, j])):
                raise ValueError("Pair potential/force is infinite at evaluated r")
            pair_potential.params[(i, j)] = dict(
                r_min=r[0], U=u[i, j][:-1], F=f[i, j][:-1]
            )
            pair_potential.r_cut[(i, j)] = r[-1]
        sim[self]["_potentials"] = [pair_potential]

        sim[self]["_bonds"] = self._get_bonds_from_snapshot(sim, snap)
        if sim.potentials.bond is not None:
            sim.bond_types = sim["engine"]["_hoomd"].state.bond_types
            bond_potential = hoomd.md.bond.Table(width=sim.potentials.bond.num)

            for i in sim.bond_types:
                r, u, f = (
                    sim.potentials.bond.linear_space,
                    sim.potentials.bond.energy(key=i),
                    sim.potentials.bond.force(key=i),
                )
                if numpy.any(numpy.isinf(u)):
                    raise ValueError("Bond potential/force is infinite at evaluated r")
                bond_potential.params[i] = dict(r_min=r[0], r_max=r[-1], U=u[:], F=f[:])
            sim[self]["_potentials"].append(bond_potential)

        sim[self]["_angles"] = self._get_angles_from_snapshot(sim, snap)
        if sim.potentials.angle is not None:
            sim.angle_types = sim["engine"]["_hoomd"].state.angle_types
            angle_potential = hoomd.md.angle.Table(width=sim.potentials.angle.num)

            for i in sim.angle_types:
                u, tau = (
                    sim.potentials.angle.energy(key=i),
                    sim.potentials.angle.force(key=i),
                )
                if numpy.any(numpy.isinf(u)):
                    raise ValueError("Angle potential/force is infinite at evaluated r")
                angle_potential.params[i] = dict(U=u[:], tau=tau[:])
            sim[self]["_potentials"].append(angle_potential)

        sim[self]["_dihedrals"] = self._get_dihedrals_from_snapshot(sim, snap)
        if sim.potentials.dihedral is not None:
            sim.dihedral_types = sim["engine"]["_hoomd"].state.dihedral_types
            dihedral_potential = hoomd.md.dihedral.Table(
                width=sim.potentials.dihedral.num
            )

            for i in sim.dihedral_types:
                u, tau = (
                    sim.potentials.dihedral.energy(key=i),
                    sim.potentials.dihedral.force(key=i),
                )
                if numpy.any(numpy.isinf(u)):
                    raise ValueError(
                        "Dihedral potential/force is infinite at evaluated r"
                    )
                dihedral_potential.params[i] = dict(U=u[:], tau=tau[:])
            sim[self]["_potentials"].append(dihedral_potential)

    @abc.abstractmethod
    def _initialize(self, sim):
        pass

    def _get_masses_from_snapshot(self, sim, snap):
        masses = collections.FixedKeyDict(sim.types)
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

    def _get_bonds_from_snapshot(self, sim, snap):
        if mpi.world.rank_is_root:
            return snap.bonds.group
        else:
            return None

    def _get_angles_from_snapshot(self, sim, snap):
        if mpi.world.rank_is_root:
            return snap.angles.group
        else:
            return None

    def _get_dihedrals_from_snapshot(self, sim, snap):
        if mpi.world.rank_is_root:
            return snap.dihedrals.group
        else:
            return None

    def _assert_dimension_safe(self, sim, snap):
        if sim.dimension == 3:
            dim_safe = True
        elif sim.dimension == 2:
            if mpi.world.rank == 0:
                dim_safe = numpy.allclose(
                    snap.particles.position[:, 2], 0
                ) and numpy.allclose(snap.particles.velocity[:, 2], 0)
            else:
                dim_safe = None
            dim_safe = mpi.world.bcast(dim_safe, root=0)
        else:
            raise ValueError("Only 2d and 3d simulations are supported")

        if not dim_safe:
            raise ValueError("Simulation initialized inconsistent with dimension")


class InitializeFromFile(InitializationOperation):
    def __init__(self, filename, format, dimension):
        self.filename = os.path.abspath(filename)
        self.format = format
        self.dimension = dimension

    def _initialize(self, sim):
        gsd_filename = self._convert_to_gsd_file(sim)
        sim["engine"]["_hoomd"].create_state_from_gsd(gsd_filename)

    def _convert_to_gsd_file(self, sim):
        file_format = initialize.InitializeFromFile._detect_format(
            self.filename, self.format
        )
        if file_format == "LAMMPS-data":
            if mpi.world.rank_is_root:
                snap = lammpsio.DataFile(self.filename).read()
                frame = snap.to_hoomd_gsd()
                frame.configuration.dimensions = self.dimension
                if self.dimension == 2:
                    frame.configuration.box[4:6] = 0.0
                    frame.configuration.box[2] = 0.0
                gsd_filename = sim.directory.temporary_file(".gsd")
                with gsd.hoomd.open(gsd_filename, _gsd_write_mode) as f:
                    f.append(frame)
            else:
                gsd_filename = None
            gsd_filename = mpi.world.bcast(gsd_filename)
        else:
            if mpi.world.rank_is_root:
                gsd_filename = self.filename
                with gsd.hoomd.open(self.filename) as trajectory:
                    frame = trajectory[0]
                    if (
                        frame.configuration.dimensions == 2
                        and frame.configuration.box[2] != 0
                    ):
                        # fix for HOOMD 2 style used in HOOMD 3
                        frame.configuration.box[2] = 0
                        gsd_filename = sim.directory.temporary_file(".gsd")
                        with gsd.hoomd.open(gsd_filename, _gsd_write_mode) as f:
                            f.append(frame)
            else:
                gsd_filename = None
            gsd_filename = mpi.world.bcast(gsd_filename)

        return gsd_filename


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

    def _initialize(self, sim):
        snap = self._make_snapshot(sim)
        sim["engine"]["_hoomd"].create_state_from_snapshot(snap)

    def _make_snapshot(self, sim):
        # make the box and snapshot
        if isinstance(self.V, extent.TriclinicBox):
            box_array = self.V.as_array("HOOMD")
            box = hoomd.Box(*box_array)
        elif isinstance(self.V, extent.ObliqueArea):
            Lx, Ly, xy = self.V.as_array("HOOMD")
            box = hoomd.Box(Lx=Lx, Ly=Ly, xy=xy)
        else:
            raise ValueError("HOOMD supports 2d and 3d simulations")

        types = tuple(self.N.keys())
        typeids = {i: typeid for typeid, i in enumerate(types)}
        snap = hoomd.Snapshot(communicator=mpi.world.comm)
        snap.configuration.box = box
        snap.particles.N = sum(self.N.values())
        snap.particles.types = list(types)

        # randomly place particles in fractional coordinates
        if mpi.world.rank == 0:
            # generate the positions and types
            if self.diameters is not None:
                (
                    positions,
                    all_types,
                ) = initialize.InitializeRandomly._pack_particles(
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


class MinimizeEnergy(simulate.SimulationOperation):
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
        self.options = options if options is not None else {}
        if "max_displacement" not in self.options:
            raise KeyError("HOOMD energy minimizer requires max_displacement option.")

    def __call__(self, sim):
        # setup FIRE minimization
        if _hoomd_version.major >= 4:
            nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
        else:
            with warnings.catch_warnings():
                if _hoomd_version >= packaging.version.Version("3.8"):
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
        steps_per_it = self.options.get("steps_per_iteration", 100)
        it = 0
        while not fire.converged and it < self.max_iterations:
            sim["engine"]["_hoomd"].run(steps_per_it)
            it += 1
        if not fire.converged:
            raise RuntimeError("Energy minimization failed to converge.")

        # cleanup
        sim["engine"]["_hoomd"].operations.integrator = None
        del fire, nve


class _Integrator(simulate.SimulationOperation):
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
        super().__init__(analyzers)
        self.steps = steps
        self.timestep = timestep

    @staticmethod
    def _make_kT(sim, thermostat, steps):
        """Cast thermostat into a kT parameter for HOOMD integrators."""
        # force the type of thermostat, in case it's a float
        if not isinstance(thermostat, md.Thermostat):
            thermostat = md.Thermostat(thermostat)

        kB = sim.potentials.kB
        if thermostat.anneal:
            if steps > 1:
                return hoomd.variant.Ramp(
                    A=kB * thermostat.T[0],
                    B=kB * thermostat.T[1],
                    t_start=sim["engine"]["_hoomd"].timestep,
                    t_ramp=steps - 1,
                )
            else:
                # if only 1 step, use the end point value. this is a corner case
                return kB * thermostat.T[1]
        else:
            return kB * thermostat.T


class RunBrownianDynamics(_Integrator):
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

    def __call__(self, sim):
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


class RunLangevinDynamics(_Integrator):
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

    def __call__(self, sim):
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


class RunMolecularDynamics(_Integrator):
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

    def __call__(self, sim):
        if self.thermostat is not None:
            kT = self._make_kT(sim, self.thermostat, self.steps)
        else:
            kT = None

        if self.thermostat is None and self.barostat is None:
            if _hoomd_version.major >= 4:
                ig_method = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
            else:
                with warnings.catch_warnings():
                    if _hoomd_version >= packaging.version.Version("3.8"):
                        warnings.simplefilter(action="ignore", category=FutureWarning)
                    ig_method = hoomd.md.methods.NVE(filter=hoomd.filter.All())
        elif (
            isinstance(self.thermostat, md.BerendsenThermostat)
            and self.barostat is None
        ):
            if _hoomd_version.major >= 4:
                ig_method = hoomd.md.methods.ConstantVolume(
                    filter=hoomd.filter.All(),
                    thermostat=hoomd.md.methods.thermostats.Berendsen(
                        kT=kT, tau=self.thermostat.tau
                    ),
                )
            else:
                with warnings.catch_warnings():
                    if _hoomd_version >= packaging.version.Version("3.8"):
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
            if _hoomd_version.major >= 4:
                ig_method = hoomd.md.methods.ConstantVolume(
                    filter=hoomd.filter.All(),
                    thermostat=hoomd.md.methods.thermostats.MTTK(
                        kT=kT, tau=self.thermostat.tau
                    ),
                )
            else:
                with warnings.catch_warnings():
                    if _hoomd_version >= packaging.version.Version("3.8"):
                        warnings.simplefilter(action="ignore", category=FutureWarning)
                    ig_method = hoomd.md.methods.NVT(
                        filter=hoomd.filter.All(),
                        kT=kT,
                        tau=self.thermostat.tau,
                    )
        elif self.thermostat is None and isinstance(self.barostat, md.MTKBarostat):
            if _hoomd_version.major >= 4:
                ig_method = hoomd.md.methods.ConstantPressure(
                    filter=hoomd.filter.All(),
                    S=self.barostat.P,
                    tauS=self.barostat.tau,
                    couple="xyz" if sim.dimension == 3 else "xy",
                )
            else:
                with warnings.catch_warnings():
                    if _hoomd_version >= packaging.version.Version("3.8"):
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
            if _hoomd_version.major >= 4:
                ig_method = hoomd.md.methods.ConstantPressure(
                    filter=hoomd.filter.All(),
                    S=self.barostat.P,
                    tauS=self.barostat.tau,
                    couple="xyz" if sim.dimension == 3 else "xy",
                    thermostat=hoomd.md.methods.thermostats.MTTK(
                        kT=kT, tau=self.thermostat.tau
                    ),
                )
            else:
                with warnings.catch_warnings():
                    if _hoomd_version >= packaging.version.Version("3.8"):
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


class EnsembleAverage(simulate.AnalysisOperation):
    def __init__(self, filename, every, rdf, assume_constraints):
        self.filename = filename
        self.every = every
        self.rdf = rdf
        self.assume_constraints = assume_constraints

    def pre_run(self, sim, sim_op):
        # thermodynamic properties
        sim[self]["_thermo"] = hoomd.md.compute.ThermodynamicQuantities(
            hoomd.filter.All()
        )
        sim[self]["_thermo_callback"] = self.EnsembleAverageAction(
            types=sim.types,
            dimension=sim.dimension,
            thermo=sim[self]["_thermo"],
            system=sim["engine"]["_hoomd"].state,
            rdf_params=self._get_rdf_params(sim),
            constraints=self._get_constrained_quantities(sim, sim_op),
            exclusions=sim.potentials.pair.exclusions,
            bonds=sim[sim.initializer]["_bonds"],
            angles=sim[sim.initializer]["_angles"],
            dihedrals=sim[sim.initializer]["_dihedrals"],
        )
        sim[self]["_hoomd_thermo_callback"] = hoomd.write.CustomWriter(
            trigger=self.every,
            action=sim[self]["_thermo_callback"],
        )
        sim["engine"]["_hoomd"].operations.computes.append(sim[self]["_thermo"])
        sim["engine"]["_hoomd"].operations.writers.append(
            sim[self]["_hoomd_thermo_callback"]
        )

    def post_run(self, sim, sim_op):
        sim["engine"]["_hoomd"].operations.writers.remove(
            sim[self]["_hoomd_thermo_callback"]
        )
        sim["engine"]["_hoomd"].operations.computes.remove(sim[self]["_thermo"])
        del sim[self]["_hoomd_thermo_callback"], sim[self]["_thermo"]

    def process(self, sim, sim_op):
        thermo_recorder = sim[self]["_thermo_callback"]
        if thermo_recorder.num_samples == 0:
            raise RuntimeError("No thermo samples were collected")
        if self.rdf is not None and thermo_recorder.num_rdf_samples == 0:
            raise RuntimeError("No RDF samples were collected")

        ens = ensemble.Ensemble(
            T=thermo_recorder.T,
            N=thermo_recorder.N,
            V=thermo_recorder.V,
            P=thermo_recorder.P,
        )
        if thermo_recorder.rdf is not None:
            for pair in thermo_recorder.rdf:
                ens.rdf[pair] = thermo_recorder.rdf[pair]

        sim[self]["ensemble"] = ens
        sim[self]["num_thermo_samples"] = thermo_recorder.num_samples
        sim[self]["num_rdf_samples"] = thermo_recorder.num_rdf_samples

        del sim[self]["_thermo_callback"]

        # optionally save file
        if self.filename is not None:
            if mpi.world.rank_is_root:
                ens.save(sim.directory.file(self.filename))
            mpi.world.barrier()

    def _get_constrained_quantities(self, sim, sim_op):
        if not self.assume_constraints:
            return None

        constraints = {}

        # then we opt-in the operations we know
        if isinstance(
            sim_op,
            (
                md.RunBrownianDynamics,
                md.RunLangevinDynamics,
                RunBrownianDynamics,
                RunLangevinDynamics,
            ),
        ):
            constraints["N"] = True
            constraints["T"] = md.Thermostat(sim_op.T)
            constraints["V"] = True
        elif isinstance(sim_op, (md.RunMolecularDynamics, RunMolecularDynamics)):
            constraints["N"] = True
            if sim_op.thermostat is not None:
                constraints["T"] = sim_op.thermostat
            # conjugate pair: one or the other is set
            if sim_op.barostat is not None:
                constraints["P"] = sim_op.barostat.P
            else:
                constraints["V"] = True

        if "T" in constraints:
            thermostat = constraints["T"]
            if thermostat.anneal:
                constraints["T"] = 0.5 * (thermostat.T[0] + thermostat.T[1])
            else:
                constraints["T"] = thermostat.T

        if "N" in constraints or "V" in constraints:

            def _get_NV_from_snapshot(sim, snap):
                box_array = None
                N = {i: 0 for i in sim.types}

                if mpi.world.rank == 0:
                    for i in sim.types:
                        N[i] = numpy.sum(
                            snap.particles.typeid == snap.particles.types.index(i)
                        )

                    box_array = snap.configuration.box
                    if sim.dimension == 2:
                        box_array = [box_array[0], box_array[1], box_array[3]]
                    box_array = numpy.array(box_array)

                N = mpi.world.bcast(N, root=0)
                box_array = mpi.world.bcast_numpy(box_array, root=0)
                if sim.dimension == 3:
                    V = extent.TriclinicBox(*box_array, convention="HOOMD")
                else:
                    V = extent.ObliqueArea(*box_array, convention="HOOMD")

                return N, V

            N = {i: 0 for i in sim.types}
            snap = sim["engine"]["_hoomd"].state.get_snapshot()
            N, V = _get_NV_from_snapshot(sim, snap)

            if "N" in constraints:
                constraints["N"] = collections.FixedKeyDict(sim.types)
                constraints["N"].update(N)

            if "V" in constraints:
                constraints["V"] = V

        # clear out dict if there are no constraints
        if len(constraints) == 0:
            constraints = None

        return constraints

    _get_rdf_params = analyze.EnsembleAverage._get_rdf_params

    class EnsembleAverageAction(Action):
        flags = [Action.Flags.PRESSURE_TENSOR]

        def __init__(
            self,
            types,
            dimension,
            thermo,
            system,
            rdf_params=None,
            constraints=None,
            exclusions=None,
            bonds=None,
            angles=None,
            dihedrals=None,
        ):
            if dimension not in (2, 3):
                raise ValueError("Only 2 or 3 dimensions supported")

            self.types = types
            self.dimension = dimension
            self.thermo = thermo
            self.system = system
            self.rdf_params = rdf_params
            self.constraints = constraints if constraints is not None else {}
            self.exclusion = exclusions
            self.bonds = bonds
            self.angles = angles
            self.dihedrals = dihedrals

            # this method handles all the initialization
            self.reset()

        def act(self, timestep):
            compute_rdf = (
                self.rdf_params is not None and timestep % self.rdf_params["every"] == 0
            )

            if "T" not in self.constraints:
                self._T += self.thermo.kinetic_temperature
            if "P" not in self.constraints:
                self._P += self.thermo.pressure
            if "V" not in self.constraints:
                for key in self._V:
                    self._V[key] += getattr(self._state.box, key)

            if "N" not in self.constraints or compute_rdf:
                snap = self.system.get_snapshot()

                if mpi.world.rank == 0:
                    # compute number of particles of each type
                    # save type masks for use in RDF calculations if needed
                    type_masks = {}
                    N = {}
                    for i in self.types:
                        type_masks[i] = (
                            snap.particles.typeid == snap.particles.types.index(i)
                        )
                        if "N" not in self.constraints:
                            N[i] = numpy.sum(type_masks[i])
                            self._N[i] += N[i]
                        else:
                            N[i] = self.constraints["N"][i]

                    # then do rdf if requested
                    if compute_rdf:
                        dimensions = snap.configuration.dimensions
                        box_array = numpy.array(snap.configuration.box)
                        box = freud.box.Box.from_box(box_array, dimensions=dimensions)

                        # calculate average density per type
                        for i in self.types:
                            self._rdf_density[i] += N[i] / box.volume
                            self._rdf_num_origins[i] += N[i]

                        # build aabb of all particles and generate a parent
                        # neighbor list with the RDF cutoff
                        aabb = freud.locality.AABBQuery(box, snap.particles.position)
                        neighbors = aabb.query(
                            snap.particles.position,
                            dict(
                                mode="ball",
                                r_max=self.rdf_params["stop"],
                                exclude_ii=True,
                            ),
                        ).toNeighborList()

                        # filter bonds from the neighbor list if they are present
                        # bond exclusions apply regardless of order, so
                        # consider both (i,j) and (j,i) permutations
                        if (
                            self.rdf_params["exclude"]
                            and snap.bonds.N != 0
                            and len(neighbors[:]) > 0
                        ):
                            bonds = numpy.vstack(
                                [self.bonds, numpy.flip(self.bonds, axis=1)],
                            )

                            bond_exclusion_filter = EnsembleAverage._cantor_pairing(
                                self, bonds, neighbors
                            )

                            neighbors.filter(bond_exclusion_filter)
                        # Similarly filter angles from the neighbor list
                        if (
                            self.rdf_params["exclude"]
                            and snap.angles.N != 0
                            and len(neighbors[:]) > 0
                        ):
                            angles = numpy.vstack(
                                [self.angles, numpy.flip(self.angles, axis=1)],
                            )

                            angle_exclusion_filter = EnsembleAverage._cantor_pairing(
                                self, angles[:, (0, -1)], neighbors
                            )

                            neighbors.filter(angle_exclusion_filter)
                        # Similarly filter dihedrals from the neighbor list
                        if (
                            self.rdf_params["exclude"]
                            and snap.dihedrals.N != 0
                            and len(neighbors[:]) > 0
                        ):
                            dihedrals = numpy.vstack(
                                [self.dihedrals, numpy.flip(self.dihedrals, axis=1)],
                            )

                            dihedral_exclusion_filter = EnsembleAverage._cantor_pairing(
                                self, dihedrals[:, (0, -1)], neighbors
                            )

                            neighbors.filter(dihedral_exclusion_filter)

                        for i in self.types:
                            for j in self.types:
                                filter_ij = numpy.logical_and(
                                    type_masks[i][neighbors[:, 0]],
                                    type_masks[j][neighbors[:, 1]],
                                )
                                counts, _ = numpy.histogram(
                                    neighbors.distances[filter_ij],
                                    bins=self.rdf_params["bins"],
                                    range=(0, self.rdf_params["stop"]),
                                )
                                self._rdf_counts[i, j] += counts
            self.num_samples += 1
            if compute_rdf:
                self.num_rdf_samples += 1

        def reset(self):
            """Resets sample number, ``T``, ``P``, and all ``V`` parameters to 0."""
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

            self._N = collections.FixedKeyDict(self.types)
            for i in self.types:
                self._N[i] = 0

            if self.rdf_params is not None:
                self._rdf_counts = {}
                self._rdf_density = {}
                self._rdf_num_origins = {}
                for i in self.types:
                    self._rdf_density[i] = 0
                    self._rdf_num_origins[i] = 0
                    for j in self.types:
                        self._rdf_counts[i, j] = numpy.zeros(
                            self.rdf_params["bins"], dtype=int
                        )
            else:
                self._rdf_counts = None
                self._rdf_density = None
                self._rdf_num_origins = None

            self.num_samples = 0
            self.num_rdf_samples = 0

        @property
        def T(self):
            """float: Average temperature across samples."""
            if "T" in self.constraints:
                return self.constraints["T"]
            else:
                if self.num_samples > 0:
                    return self._T / self.num_samples
                else:
                    return None

        @property
        def P(self):
            """float: Average pressure across samples."""
            if "P" in self.constraints:
                return self.constraints["P"]
            else:
                if self.num_samples > 0:
                    return self._P / self.num_samples
                else:
                    return None

        @property
        def V(self):
            """float: Average extent across samples."""
            if "V" in self.constraints:
                return self.constraints["V"]
            else:
                if self.num_samples > 0:
                    _V = {key: self._V[key] / self.num_samples for key in self._V}
                    if self.dimension == 3:
                        return extent.TriclinicBox(**_V, convention="HOOMD")
                    elif self.dimension == 2:
                        return extent.ObliqueArea(**_V, convention="HOOMD")
                else:
                    return None

        @property
        def N(self):
            """:class:`~relentless.collections.FixedKeyDict`:
            Number of particles by type.
            """
            if "N" in self.constraints:
                return self.constraints["N"]
            else:
                if self.num_samples > 0:
                    N = collections.FixedKeyDict(self._N.keys())
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
            if self.num_rdf_samples > 0:
                bin_edges = numpy.linspace(
                    0, self.rdf_params["stop"], self.rdf_params["bins"] + 1
                )
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                if self.dimension == 3:
                    bin_extents = (4 * numpy.pi / 3) * (
                        bin_edges[1:] ** 3 - bin_edges[:-1] ** 3
                    )
                elif self.dimension == 2:
                    bin_extents = numpy.pi * (bin_edges[1:] ** 2 - bin_edges[:-1] ** 2)

                rdf = collections.PairMatrix(self.types)
                for i, j in rdf:
                    if mpi.world.rank == 0:
                        density = {
                            k: self._rdf_density[k] / self.num_rdf_samples
                            for k in self.types
                        }
                        g = numpy.zeros_like(bin_centers)
                        if i == j:
                            if self._rdf_num_origins[i] > 0 and density[i] > 0:
                                g = self._rdf_counts[i, i] / (
                                    self._rdf_num_origins[i] * density[i] * bin_extents
                                )
                        else:
                            # this takes the weighted average of g_ij and g_ji
                            num_ij_origins = (
                                self._rdf_num_origins[i] + self._rdf_num_origins[j]
                            )
                            if num_ij_origins > 0:
                                if density[j] > 0:
                                    g += self._rdf_counts[i, j] / (
                                        num_ij_origins * density[j] * bin_extents
                                    )
                                if density[i] > 0:
                                    g += self._rdf_counts[j, i] / (
                                        num_ij_origins * density[i] * bin_extents
                                    )
                        gr = numpy.column_stack((bin_centers, g))
                    else:
                        gr = None
                    gr = mpi.world.bcast_numpy(gr, root=0)
                    rdf[i, j] = ensemble.RDF(gr[:, 0], gr[:, 1])
                return rdf
            else:
                return None


class Record(simulate.AnalysisOperation):
    def __init__(self, filename, every, quantities):
        self.filename = filename
        self.every = every
        self.quantities = quantities

    def pre_run(self, sim, sim_op):
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

    def post_run(self, sim, sim_op):
        sim["engine"]["_hoomd"].operations.writers.remove(sim[self]["_hoomd_recorder"])
        sim["engine"]["_hoomd"].operations.computes.remove(sim[self]["_thermo"])
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

        # optionally save file
        if self.filename is not None:
            if mpi.world.rank_is_root:
                analyze.Record._save(
                    sim.directory.file(self.filename), self.quantities, sim[self]
                )
            mpi.world.barrier()

    class RecorderCallback(Action):
        flags = [Action.Flags.PRESSURE_TENSOR]

        def __init__(self, thermo, quantities):
            self.thermo = thermo
            self.quantities = quantities
            self.reset()

        def act(self, timestep):
            self.timestep.append(timestep)
            for q in self.quantities:
                if q == "temperature":
                    val = getattr(self.thermo, "kinetic_temperature")
                else:
                    val = getattr(self.thermo, q)
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


class WriteTrajectory(simulate.AnalysisOperation):
    def __init__(self, filename, every, format, velocities, images, types, masses):
        self.filename = filename
        self.every = every
        self.format = format
        self.velocities = velocities
        self.images = images
        self.types = types
        self.masses = masses

    def pre_run(self, sim, sim_op):
        sim[self]["_gsd"] = hoomd.write.GSD(
            trigger=self.every,
            filename=sim.directory.file(self.filename),
            filter=hoomd.filter.All(),
            mode="wb",
            dynamic=self._get_dynamic(),
        )
        sim["engine"]["_hoomd"].operations.writers.append(sim[self]["_gsd"])

    def post_run(self, sim, sim_op):
        if _hoomd_version.major >= 4:
            sim[self]["_gsd"].flush()
        sim["engine"]["_hoomd"].operations.writers.remove(sim[self]["_gsd"])
        del sim[self]["_gsd"]

    def process(self, sim, sim_op):
        filename = sim.directory.file(self.filename)
        file_format = analyze.WriteTrajectory._detect_format(filename, self.format)
        if file_format == "LAMMPS-dump":
            if mpi.world.rank_is_root:
                dump_file = sim.directory.temporary_file(".dump")
                snaps = []
                with gsd.hoomd.open(filename) as t:
                    for s in t:
                        snap, _ = lammpsio.Snapshot.from_hoomd_gsd(s)
                        snaps.append(snap)

                schema = analyze.WriteTrajectory._make_lammps_schema(
                    self.velocities, self.images, self.types, self.masses
                )
                lammpsio.DumpFile.create(dump_file, schema, snaps)
                shutil.move(dump_file, filename)
            mpi.world.barrier()

    def _get_dynamic(self):
        # property group is always made dyanmic since it logs box and position
        dynamic = ["property"]

        if _hoomd_version.major >= 4:
            if self.velocities is True:
                dynamic.append("particles/velocity")
            if self.images is True:
                dynamic.append("particles/images")
        else:
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
    HOOMD 3.x and 4.x are supported.

    .. warning::

        HOOMD requires that tabulated pair potentials be finite. A common place to
        have an infinite value is at :math:`r=0`, since potentials like
        :class:`~relentless.potential.pair.LennardJones` diverge there. You should
        make sure to exclude these values from the tabulated potentials,
        e.g., setting :attr:`~relentless.simulate.PairPotentialTabulator.rmin` to a
        small value larger than 0.

    Parameters
    ----------
    initializer : :class:`~relentless.simulate.SimulationOperation`
        Operation that initializes the simulation.
    operations : array_like
        :class:`~relentless.simulate.SimulationOperation` to execute for run.
        Defaults to ``None``, which means nothing is done after initialization.
    device : str
        Device to execute on. Values are "cpu" (run on CPU), "gpu" (run on GPU),
        and "auto" (default, select best available).

    Raises
    ------
    ImportError
        If the :mod:`hoomd` package is not found or is not version 2.x.

    """

    def __init__(self, initializer, operations=None, device="auto"):
        if not _hoomd_found:
            raise ImportError("HOOMD not found.")

        if _hoomd_version.major < 3:
            raise ImportError("HOOMD version 3 or later is required")

        super().__init__(initializer, operations)
        self.device = device

    def _initialize_engine(self, sim):
        sim["engine"]["version"] = _hoomd_version
        if self.device == "auto":
            device_class = hoomd.device.auto_select
        elif self.device.lower() == "cpu":
            device_class = hoomd.device.CPU
        elif self.device.lower() == "gpu":
            device_class = hoomd.device.GPU
        else:
            raise ValueError("Unrecognized device type, must be in (auto, cpu, gpu)")
        device = device_class(communicator=mpi.world.comm, notice_level=0)
        sim["engine"]["_hoomd"] = hoomd.Simulation(device)

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
