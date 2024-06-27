import abc
import datetime
import os
import shutil
import subprocess

import freud
import gsd.hoomd
import lammpsio
import numpy
import packaging.version

from relentless import collections, mpi
from relentless.model import ensemble, extent

from . import analyze, initialize, md, simulate

try:
    import lammps

    _lammps_found = True
except ImportError:
    _lammps_found = False

try:
    _gsd_version = gsd.version.version
except AttributeError:
    _gsd_version = gsd.__version__
if packaging.version.Version(_gsd_version) >= packaging.version.Version("2.8.0"):
    _gsd_write_mode = "w"
else:
    _gsd_write_mode = "wb"


class Counters:
    _compute = 1
    _dump = 1
    _fix = 1
    _group = 1
    _variable = 1

    @classmethod
    def new_compute_id(cls):
        """Make a unique new compute ID.

        Returns
        -------
        int
            The compute ID.

        """
        idx = int(cls._compute)
        cls._compute += 1
        return "c{}".format(idx)

    @classmethod
    def new_dump_id(cls):
        """Make a unique new dump ID.

        Returns
        -------
        int
            The dump ID.

        """
        idx = int(cls._dump)
        cls._dump += 1
        return "d{}".format(idx)

    @classmethod
    def new_fix_id(cls):
        """Make a unique new fix ID.

        Returns
        -------
        int
            The fix ID.

        """
        idx = int(cls._fix)
        cls._fix += 1
        return "f{}".format(idx)

    @classmethod
    def new_group_id(cls):
        """Make a unique new fix ID.

        Returns
        -------
        int
            The fix ID.

        """
        idx = int(cls._group)
        cls._group += 1
        return "g{}".format(idx)

    @classmethod
    def new_variable_id(cls):
        """Make a unique new variable ID.

        Returns
        -------
        int
            The variable ID.

        """
        idx = int(cls._variable)
        cls._variable += 1
        return "v{}".format(idx)


class SimulationOperation(simulate.SimulationOperation):
    """LAMMPS simulation operation."""

    def __call__(self, sim):
        """Evaluate the LAMMPS simulation operation.

        Each deriving class of :class:`SimulationOperation` must implement a
        :meth:`_call_commands()` method that returns a list or tuple of LAMMPS
        commands that can be executed by :meth:`lammps.commands_list()`.

        Parameters
        ----------
        sim : :class:`~relentless.simulate.SimulationInstance`
            The simulation instance.

        """
        cmds = self._call_commands(sim)
        if cmds is None or len(cmds) == 0:
            return

        sim["engine"]["_lammps_commands"] += cmds
        if sim["engine"]["use_python"]:
            sim["engine"]["_lammps"].commands_list(cmds)

    @abc.abstractmethod
    def _call_commands(self, sim):
        """Create the LAMMPS commands for the simulation operation.

        All deriving classes must implement this method.

        Parameters
        ----------
        sim : :class:`~relentless.simulate.SimulationInstance`
            The simulation instance.

        Returns
        -------
        array_like
            The LAMMPS commands for this operation.

        """
        pass


# initializers
class InitializationOperation(SimulationOperation, simulate.InitializationOperation):
    """Initialize a simulation."""

    def _call_commands(self, sim):
        cmds = [
            "units {style}".format(style=sim["engine"]["units"]),
            "boundary p p p",
            "atom_style {style}".format(style=sim["engine"]["atom_style"]),
        ]
        cmds += self.initialize_commands(sim)
        # dimension is only available after initialization commands, so we insert it now
        cmds.insert(2, f"dimension {sim.dimension}")
        sim.types = sim["engine"]["types"].keys()

        # create a group for each type
        sim[self]["_type_groups"] = collections.FixedKeyDict(sim.types)
        for i in sim.types:
            typeid = sim["engine"]["types"][i]
            groupid = Counters.new_group_id()
            sim[self]["_type_groups"][i] = groupid
            cmds.append(f"group {groupid} type {typeid}")

        # process from data file
        sim.masses = collections.FixedKeyDict(sim.types)
        masses = {}
        dim_safe = None
        if mpi.world.rank_is_root:
            snap = lammpsio.DataFile(sim[self]["_datafile"]).read()
            for i in sim.types:
                mi = snap.mass[snap.typeid == sim["engine"]["types"][i]]
                if len(mi) == 0:
                    raise KeyError("Type {} not present in simulation".format(i))
                elif not numpy.all(mi == mi[0]):
                    raise ValueError("All masses for a type must be equal")
                masses[i] = mi[0]

            if sim.dimension == 3:
                dim_safe = True
            elif sim.dimension == 2:
                dim_safe = numpy.allclose(snap.position[:, 2], 0) and numpy.allclose(
                    snap.velocity[:, 2], 0
                )
            else:
                raise ValueError("Only 2d and 3d simulations are supported")
        masses = mpi.world.bcast(masses)
        sim.masses.update(masses)
        dim_safe = mpi.world.bcast(dim_safe)
        if not dim_safe:
            raise ValueError("Simulation initialized inconsistent with dimension")

        # attach the potentials
        if sim.potentials.pair.start == 0:
            raise ValueError("LAMMPS requires start > 0 for pair potentials")
        r, u, f = sim.potentials.pair.pairwise_energy_and_force(
            sim.types, x=sim.potentials.pair.squared_space, tight=True, minimum_num=2
        )
        Nr = len(r)
        sim[self]["_potentials_rmax"] = r[-1]

        def pair_map(sim, pair):
            # Map lammps type indexes as a pair, lowest type first
            i, j = pair
            id_i = sim["engine"]["types"][i]
            id_j = sim["engine"]["types"][j]
            if id_i > id_j:
                id_i, id_j = id_j, id_i

            return id_i, id_j

        # write all potentials into a file
        if mpi.world.rank_is_root:
            file_ = sim.directory.temporary_file()
            with open(file_, "w") as fw:
                fw.write("# LAMMPS tabulated pair potentials\n")
                for i, j in sim.pairs:
                    if numpy.any(numpy.isinf(u[i, j])) or numpy.any(
                        numpy.isinf(f[i, j])
                    ):
                        raise ValueError(
                            "Pair potential/force is infinite at evaluated r"
                        )

                    id_i, id_j = pair_map(sim, (i, j))
                    fw.write(
                        ("# pair ({i},{j})\n" "\n" "TABLE_{id_i}_{id_j}\n").format(
                            i=i, j=j, id_i=id_i, id_j=id_j
                        )
                    )
                    fw.write(
                        "N {N} RSQ {rmin} {rmax}\n\n".format(
                            N=Nr,
                            rmin=r[0],
                            rmax=r[-1],
                        )
                    )

                    for idx, (r_, u_, f_) in enumerate(
                        zip(r, u[i, j], f[i, j]), start=1
                    ):
                        fw.write(
                            "{idx} {r} {u} {f}\n".format(idx=idx, r=r_, u=u_, f=f_)
                        )
        else:
            file_ = None
        file_ = mpi.world.bcast(file_)

        # process all lammps commands
        cmds += [
            "neighbor {skin} multi".format(skin=sim.potentials.pair.neighbor_buffer),
            "neigh_modify delay 0 every 1 check yes",
        ]
        cmds += ["pair_style table linear {N}".format(N=Nr)]

        for i, j in sim.pairs:
            # get lammps type indexes, lowest type first
            id_i, id_j = pair_map(sim, (i, j))
            cmds += [
                ("pair_coeff {id_i} {id_j} {filename}" " TABLE_{id_i}_{id_j}").format(
                    id_i=id_i, id_j=id_j, filename=file_
                )
            ]

        return cmds

    @abc.abstractmethod
    def initialize_commands(self, sim):
        pass


class InitializeFromFile(InitializationOperation):
    def __init__(self, filename, format, dimension):
        self.filename = os.path.abspath(filename)
        self.format = format
        self.dimension = dimension

    def initialize_commands(self, sim):
        data_filename, dimension, type_map = self._convert_to_data_file(sim)
        sim[self]["_datafile"] = data_filename
        if sim["engine"]["types"] is None:
            sim["engine"]["types"] = type_map
        elif sim["engine"]["types"] != type_map:
            raise ValueError("Specified LAMMPS type map does not match detected map")
        sim.dimension = dimension
        return [f"read_data {data_filename}"]

    def _convert_to_data_file(self, sim):
        file_format = initialize.InitializeFromFile._detect_format(
            self.filename, self.format
        )
        if file_format == "HOOMD-GSD":
            if mpi.world.rank_is_root:
                data_filename = sim.directory.temporary_file(".data")
                with gsd.hoomd.open(self.filename) as f:
                    frame = f[0]
                snap, type_map = lammpsio.Snapshot.from_hoomd_gsd(frame)
                type_map = {v: k for k, v in type_map.items()}

                # figure out dimensions
                dimension = self.dimension
                if dimension is None:
                    dimension = frame.configuration.dimensions
                elif dimension != frame.configuration.dimensions:
                    raise ValueError("Specified dimension does not match GSD dimension")

                # fix up 2d boxes that may not be compatible with lammps
                if dimension == 2:
                    if frame.configuration.box[2] == 0:
                        snap.low[2] = -0.5
                        snap.high[2] = 0.5
                    if snap.box.tilt is not None:
                        snap.box.tilt[1:] = 0.0
                lammpsio.DataFile.create(
                    data_filename, snap, sim["engine"]["atom_style"]
                )
            else:
                data_filename = None
                dimension = None
                type_map = None
            data_filename = mpi.world.bcast(data_filename)
            dimension = mpi.world.bcast(dimension)
            type_map = mpi.world.bcast(type_map)
        else:
            data_filename = self.filename
            dimension = self.dimension if self.dimension is not None else 3

            type_map = sim["engine"]["types"]
            if type_map is None:
                if mpi.world.rank_is_root:
                    snap = lammpsio.DataFile(
                        data_filename, sim["engine"]["atom_style"]
                    ).read()
                    typeids = numpy.unique(snap.typeid)
                    type_map = {str(typeid): typeid for typeid in typeids}
                else:
                    type_map = None
                type_map = mpi.world.bcast(type_map)

        return data_filename, dimension, type_map


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

    def initialize_commands(self, sim):
        if isinstance(self.V, extent.TriclinicBox):
            sim.dimension = 3
        elif isinstance(self.V, extent.ObliqueArea):
            sim.dimension = 2
        else:
            raise TypeError(
                "LAMMPS boxes must be derived from TriclinicBox or ObliqueArea"
            )

        if sim["engine"]["types"] is None:
            sim["engine"]["types"] = {i: idx + 1 for idx, i in enumerate(self.N.keys())}

        if mpi.world.rank_is_root:
            # make box
            if sim.dimension == 3:
                Lx, Ly, Lz, xy, xz, yz = self.V.as_array("LAMMPS")
                lo = self.V.low
            elif sim.dimension == 2:
                Lx, Ly, xy = self.V.as_array("LAMMPS")
                Lz = 1.0
                xz = 0.0
                yz = 0.0
                lo = numpy.array([self.V.low[0], self.V.low[1], -0.5 * Lz])
            else:
                raise ValueError("LAMMPS only supports 2d and 3d simulations")
            hi = lo + [Lx, Ly, Lz]
            tilt = numpy.array([xy, xz, yz])
            if not numpy.all(numpy.isclose(tilt, 0)):
                box = lammpsio.Box(lo, hi, tilt)
            else:
                box = lammpsio.Box(lo, hi)
            snap = lammpsio.Snapshot(N=sum(self.N.values()), box=box)

            # generate the positions and types
            if self.diameters is not None:
                positions, all_types = initialize.InitializeRandomly._pack_particles(
                    self.seed, self.N, self.V, self.diameters
                )
            else:
                positions, all_types = initialize.InitializeRandomly._random_particles(
                    self.seed, self.N, self.V
                )

            # set the positions
            snap.position[:, : sim.dimension] = positions

            # set the typeids
            snap.typeid = [sim["engine"]["types"][i] for i in all_types]

            # set masses, defaulting to unit mass
            if self.masses is not None:
                snap.mass = [self.masses[i] for i in all_types]
            else:
                snap.mass[:] = 1.0

            # set velocities, defaulting to zeros
            if self.T is not None:
                rng = numpy.random.default_rng(self.seed + 1)
                # Maxwell-Boltzmann = normal with variance kT/m per component
                vel = rng.normal(
                    scale=numpy.sqrt(sim.potentials.kB * self.T),
                    size=(snap.N, sim.dimension),
                )
                vel /= numpy.sqrt(snap.mass[:, None])

                # zero the linear momentum
                v_cm = numpy.sum(snap.mass[:, None] * vel, axis=0) / numpy.sum(
                    snap.mass
                )
                vel -= v_cm
            else:
                vel = numpy.zeros((snap.N, sim.dimension))
            snap.velocity[:, : sim.dimension] = vel

            init_file = sim.directory.temporary_file(".data")
            lammpsio.DataFile.create(init_file, snap, sim["engine"]["atom_style"])
        else:
            init_file = None
        init_file = mpi.world.bcast(init_file)

        sim[self]["_datafile"] = init_file
        return ["read_data {}".format(init_file)]


# simulation operations
class MinimizeEnergy(SimulationOperation):
    """Perform energy minimization on a configuration.

    Valid **options** include:

    - **max_evaluations** (`int`) - the maximum number of force/energy evaluations.
      Defaults to ``100*max_iterations``.

    Parameters
    ----------
    energy_tolerance : float
        Energy convergence criterion.
    force_tolerance : float
        Force convergence criterion.
    max_iterations : int
        Maximum number of iterations to run the minimization.
    options : dict
        Additional options for energy minimzer.

    """

    def __init__(self, energy_tolerance, force_tolerance, max_iterations, options):
        self.energy_tolerance = energy_tolerance
        self.force_tolerance = force_tolerance
        self.max_iterations = max_iterations
        self.options = options if options is not None else {}

    def _call_commands(self, sim):
        max_eval = self.options.get("max_evaluations", 100 * self.max_iterations)
        cmds = [
            "minimize {etol} {ftol} {maxiter} {maxeval}".format(
                etol=self.energy_tolerance,
                ftol=self.force_tolerance,
                maxiter=self.max_iterations,
                maxeval=max_eval,
            )
        ]
        if sim.dimension == 2:
            fix_2d = Counters.new_fix_id()
            cmds = (
                ["fix {} all enforce2d".format(fix_2d)]
                + cmds
                + ["unfix {}".format(fix_2d)]
            )

        return cmds


class _Integrator(SimulationOperation):
    """Base LAMMPS molecular dynamics integrator.

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

    def __call__(self, sim):
        for analyzer in self.analyzers:
            analyzer.pre_run(sim, self)

        super().__call__(sim)

        for analyzer in self.analyzers:
            analyzer.post_run(sim, self)

    def _run_commands(self, sim):
        cmds = ["run {N}".format(N=self.steps)]

        # wrap with fixes
        if sim.dimension == 2:
            fix_2d = Counters.new_fix_id()
            cmds = (
                ["fix {} all enforce2d".format(fix_2d)]
                + cmds
                + ["unfix {}".format(fix_2d)]
            )

        return cmds

    @staticmethod
    def _make_T(thermostat):
        """Cast thermostat into a T parameter for LAMMPS integrators."""
        # force the type of thermostat, in case it's a float
        if not isinstance(thermostat, md.Thermostat):
            thermostat = md.Thermostat(thermostat)

        if thermostat.anneal:
            return (thermostat.T[0], thermostat.T[1])
        else:
            return (thermostat.T, thermostat.T)


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

    def _call_commands(self, sim):
        if "BROWNIAN" not in sim["engine"]["packages"]:
            raise NotImplementedError("LAMMPS BROWNIAN package is not installed.")
        elif sim["engine"]["version"] < 20220623:
            raise NotImplementedError(
                "LAMMPS versions prior to 23Jun2022 stable release do not"
                " properly support Brownian dynamics."
            )

        T = self._make_T(self.T)
        if T[0] != T[1]:
            raise NotImplementedError(
                "Brownian dynamics cannot do temperature annealing in LAMMPS."
            )

        try:
            len(self.friction)
            same_friction = False
        except TypeError:
            same_friction = True

        fix_ids = []
        cmd_template = "fix {fixid} {groupid} brownian {T} {seed} gamma_t {friction}"

        cmds = ["timestep {}".format(self.timestep)]
        if same_friction:
            fixid = Counters.new_fix_id()
            cmds.append(
                cmd_template.format(
                    fixid=fixid,
                    groupid="all",
                    T=T[0],
                    seed=self.seed,
                    friction=self.friction,
                )
            )
            fix_ids.append(fixid)
        else:
            for i, t in enumerate(sim.types):
                groupid = sim[sim.initializer]["_type_groups"][t]
                fixid = Counters.new_fix_id()
                cmds.append(
                    cmd_template.format(
                        fixid=fixid,
                        groupid=groupid,
                        T=T[0],
                        seed=self.seed + i,
                        friction=self.friction[t],
                    ),
                )
                fix_ids.append(fixid)
        cmds += self._run_commands(sim)
        cmds += ["unfix {}".format(idx) for idx in fix_ids]
        return cmds


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

    def _call_commands(self, sim):
        # obtain per-type friction factor
        Ntypes = len(sim.types)
        mass = numpy.zeros(Ntypes)
        friction = numpy.zeros(Ntypes)
        for t in sim.types:
            typeidx = sim["engine"]["types"][t] - 1
            try:
                friction[typeidx] = self.friction[t]
            except TypeError:
                friction[typeidx] = self.friction
            except KeyError:
                raise KeyError("The friction factor for type {} is not set.".format(t))

            mass[typeidx] = sim.masses[t]

        # compute per-type damping parameter and rescale if multiple types
        damp = numpy.divide(mass, friction, where=(friction > 0))
        damp_ref = damp[0]
        if Ntypes > 1:
            scale = damp / damp_ref
            scale_str = " ".join(
                ["scale {} {}".format(i + 2, s) for i, s in enumerate(scale[1:])]
            )
        else:
            scale_str = ""

        T = self._make_T(self.T)
        fix_ids = {"nve": Counters.new_fix_id(), "langevin": Counters.new_fix_id()}
        cmds = [
            "timestep {}".format(self.timestep),
            "fix {idx} {group_idx} nve".format(idx=fix_ids["nve"], group_idx="all"),
            (
                "fix {idx} {group_idx} langevin {t_start} {t_stop}"
                " {damp} {seed} {scaling}"
            ).format(
                idx=fix_ids["langevin"],
                group_idx="all",
                t_start=T[0],
                t_stop=T[1],
                damp=damp_ref,
                seed=self.seed,
                scaling=scale_str,
            ),
        ]
        cmds += self._run_commands(sim)
        cmds += ["unfix {}".format(idx) for idx in fix_ids.values()]
        return cmds


class RunMolecularDynamics(_Integrator):
    """Perform a molecular dynamics simulation.

    This method supports:

    - NVE integration
    - NVT integration with Nosé-Hoover or Berendsen thermostat
    - NPH integration with MTK or Berendsen barostat
    - NPT integration with Nosé-Hoover or Berendsen thermostat and
      MTK or Berendsen barostat

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

    def _call_commands(self, sim):
        fix_ids = {"ig": Counters.new_fix_id()}

        if self.thermostat is not None:
            T = self._make_T(self.thermostat)
        else:
            T = None

        cmds = ["timestep {}".format(self.timestep)]
        if (
            self.thermostat is None
            or isinstance(self.thermostat, md.BerendsenThermostat)
        ) and (
            self.barostat is None or isinstance(self.barostat, md.BerendsenBarostat)
        ):
            cmds += [
                "fix {idx} {group_idx} nve".format(idx=fix_ids["ig"], group_idx="all")
            ]
        elif isinstance(self.thermostat, md.NoseHooverThermostat) and (
            self.barostat is None or isinstance(self.barostat, md.BerendsenBarostat)
        ):
            cmds += [
                "fix {idx} {group_idx} nvt temp {Tstart} {Tstop} {Tdamp}".format(
                    idx=fix_ids["ig"],
                    group_idx="all",
                    Tstart=T[0],
                    Tstop=T[1],
                    Tdamp=self.thermostat.tau,
                )
            ]
        elif (
            self.thermostat is None
            or isinstance(self.thermostat, md.BerendsenThermostat)
        ) and isinstance(self.barostat, md.MTKBarostat):
            cmds += [
                "fix {idx} {group_idx} nph iso {Pstart} {Pstop} {Pdamp}".format(
                    idx=fix_ids["ig"],
                    group_idx="all",
                    Pstart=self.barostat.P,
                    Pstop=self.barostat.P,
                    Pdamp=self.barostat.tau,
                )
            ]
        elif isinstance(self.thermostat, md.NoseHooverThermostat) and isinstance(
            self.barostat, md.MTKBarostat
        ):
            cmds += [
                (
                    "fix {idx} {group_idx} npt temp {Tstart} {Tstop} {Tdamp}"
                    " iso {Pstart} {Pstop} {Pdamp}"
                ).format(
                    idx=fix_ids["ig"],
                    group_idx="all",
                    Tstart=T[0],
                    Tstop=T[1],
                    Tdamp=self.thermostat.tau,
                    Pstart=self.barostat.P,
                    Pstop=self.barostat.P,
                    Pdamp=self.barostat.tau,
                )
            ]
        else:
            raise TypeError(
                "An appropriate combination of thermostat and barostat must be set."
            )

        if isinstance(self.thermostat, md.BerendsenThermostat):
            fix_ids["berendsen_temp"] = Counters.new_fix_id()
            cmds += [
                "fix {idx} {group_idx} temp/berendsen {Tstart} {Tstop} {Tdamp}".format(
                    idx=fix_ids["berendsen_temp"],
                    group_idx="all",
                    Tstart=T[0],
                    Tstop=T[1],
                    Tdamp=self.thermostat.tau,
                )
            ]
        if isinstance(self.barostat, md.BerendsenBarostat):
            fix_ids["berendsen_press"] = Counters.new_fix_id()
            cmds += [
                (
                    "fix {idx} {group_idx} press/berendsen iso"
                    " {Pstart} {Pstop} {Pdamp}"
                ).format(
                    idx=fix_ids["berendsen_press"],
                    group_idx="all",
                    Pstart=self.barostat.P,
                    Pstop=self.barostat.P,
                    Pdamp=self.barostat.tau,
                )
            ]

        cmds += self._run_commands(sim)
        cmds += ["unfix {}".format(idx) for idx in fix_ids.values()]
        return cmds


# analyzers
class AnalysisOperation(simulate.AnalysisOperation):
    def pre_run(self, sim, sim_op):
        cmds = self._pre_run_commands(sim, sim_op)
        if cmds is None or len(cmds) == 0:
            return

        sim["engine"]["_lammps_commands"] += cmds
        if sim["engine"]["use_python"]:
            sim["engine"]["_lammps"].commands_list(cmds)

    def post_run(self, sim, sim_op):
        cmds = self._post_run_commands(sim, sim_op)
        if cmds is None or len(cmds) == 0:
            return

        sim["engine"]["_lammps_commands"] += cmds
        if sim["engine"]["use_python"]:
            sim["engine"]["_lammps"].commands_list(cmds)

    @abc.abstractmethod
    def _pre_run_commands(self, sim, sim_op):
        pass

    @abc.abstractmethod
    def _post_run_commands(self, sim, sim_op):
        pass


class EnsembleAverage(AnalysisOperation):
    def __init__(self, filename, every, rdf, assume_constraints):
        self.filename = filename
        self.every = every
        self.rdf = rdf
        self.assume_constraints = assume_constraints

    def _pre_run_commands(self, sim, sim_op):
        cmds = []

        constraints = self._get_constrained_quantities(sim, sim_op)
        if constraints is None:
            constraints = {}
        sim[self]["_constraints"] = constraints

        # dicts are insertion ordered, so we can rely on this for thermo columns
        fix_ids = {}
        var_ids = {}
        if "T" not in constraints:
            var_ids["T"] = Counters.new_variable_id()
            cmds.append("variable {} equal temp".format(var_ids["T"]))
        if "P" not in constraints:
            var_ids["P"] = Counters.new_variable_id()
            cmds.append("variable {} equal press".format(var_ids["P"]))
        if "V" not in constraints:
            var_ids.update(
                {
                    "Lx": Counters.new_variable_id(),
                    "Ly": Counters.new_variable_id(),
                    "Lz": Counters.new_variable_id(),
                    "xy": Counters.new_variable_id(),
                    "xz": Counters.new_variable_id(),
                    "yz": Counters.new_variable_id(),
                }
            )
            cmds += [
                "variable {} equal lx".format(var_ids["Lx"]),
                "variable {} equal ly".format(var_ids["Ly"]),
                "variable {} equal lz".format(var_ids["Lz"]),
                "variable {} equal xy".format(var_ids["xy"]),
                "variable {} equal xz".format(var_ids["xz"]),
                "variable {} equal yz".format(var_ids["yz"]),
            ]
        if "N" not in constraints:
            for i in sim.types:
                groupid = sim[sim.initializer]["_type_groups"][i]
                typekey = f"N_{i}"
                var_ids[typekey] = Counters.new_variable_id()
                cmds.append(
                    'variable {vid} equal "count({gid})"'.format(
                        vid=var_ids[typekey], gid=groupid
                    ),
                )

        # generate temporary file names, may or may not get used but that's OK
        if mpi.world.rank_is_root:
            file_ = {
                "thermo": sim.directory.temporary_file(),
                "rdf": sim.directory.temporary_file(".data"),
                "data": sim.directory.temporary_file(".data"),
            }
        else:
            file_ = None
        file_ = mpi.world.bcast(file_)

        # thermodynamic properties
        if len(var_ids) > 0:
            sim[self]["_thermo_file"] = file_["thermo"]
            fix_ids["thermo_avg"] = Counters.new_fix_id()
            cmds += [
                (
                    "fix {fixid} all ave/time {every} 1 {every}"
                    + " "
                    + " ".join(["v_" + v_ for v_ in var_ids.values()])
                    + " mode scalar ave running"
                    ' file {filename} overwrite format " %.16e"'
                ).format(
                    fixid=fix_ids["thermo_avg"],
                    every=self.every,
                    filename=sim[self]["_thermo_file"],
                )
            ]

            sim[self]["_thermo_columns"] = {k: i for i, k in enumerate(var_ids.keys())}

        # write a data file if there are constraints on N or V
        if "N" in constraints or "V" in constraints:
            cmds += ["write_data {filename}".format(filename=file_["data"])]
            sim[self]["_constraint_data_file"] = file_["data"]

        # dump a trajectory for the RDF calculation
        rdf_params = self._get_rdf_params(sim)
        if rdf_params is not None:
            sim[self]["_rdf_params"] = rdf_params
            sim[self]["_rdf_file"] = file_["rdf"]
            sim[self]["_rdf_dump"] = WriteTrajectory(
                filename=sim[self]["_rdf_file"],
                every=rdf_params["every"],
                format="LAMMPS-dump",
                velocities=False,
                images=False,
                types=True,
                masses=False,
            )
            sim[self]["_rdf_dump"].pre_run(sim, sim_op)

        # save ids so we can remove them later
        sim[self]["_fix_ids"] = fix_ids
        sim[self]["_var_ids"] = var_ids

        return cmds

    def _post_run_commands(self, sim, sim_op):
        cmds = []
        # unfix
        for fixid in sim[self]["_fix_ids"].values():
            cmds.append("unfix {}".format(fixid))
        del sim[self]["_fix_ids"]
        # delete variables
        for var_id in sim[self]["_var_ids"].values():
            cmds.append("variable {} delete".format(var_id))
        del sim[self]["_var_ids"]

        if "_rdf_dump" in sim[self]:
            sim[self]["_rdf_dump"].post_run(sim, sim_op)

        return cmds

    def process(self, sim, sim_op):
        # first process any constraints
        constraints = sim[self]["_constraints"]
        if "N" in constraints or "V" in constraints:
            if mpi.world.rank_is_root:
                snap = lammpsio.DataFile(sim[self]["_constraint_data_file"]).read()
            else:
                snap = None

            if "N" in constraints:
                if mpi.world.rank_is_root:
                    N = {
                        i: numpy.sum(snap.typeid == sim["engine"]["types"][i])
                        for i in sim.types
                    }
                else:
                    N = None
                N = mpi.world.bcast(N)

            if "V" in constraints:
                if mpi.world.rank_is_root:
                    L = snap.box.high - snap.box.low
                    L = L[: sim.dimension]
                    tilt = snap.box.tilt
                    if tilt is None:
                        if sim.dimension == 3:
                            tilt = [0, 0, 0]
                        else:
                            tilt = [0]
                    box_array = numpy.concatenate((L, tilt))
                else:
                    box_array = None

                box_array = mpi.world.bcast_numpy(box_array)
                if sim.dimension == 3:
                    V = extent.TriclinicBox(*box_array, convention="LAMMPS")
                else:
                    V = extent.ObliqueArea(*box_array, convention="LAMMPS")

        # extract thermo properties
        # we skip the first 2 rows, which are LAMMPS junk, and slice out the
        # timestep from col. 0
        try:
            thermo = mpi.world.loadtxt(sim[self]["_thermo_file"], skiprows=2)[1:]
        except Exception as e:
            raise RuntimeError("No LAMMPS thermo file generated") from e
        columns = sim[self]["_thermo_columns"]

        if "T" not in constraints:
            T = thermo[columns["T"]]
        else:
            T = constraints["T"]

        if "P" not in constraints:
            P = thermo[columns["P"]]
        else:
            P = constraints["P"]

        if "N" not in constraints:
            N = {i: thermo[columns[f"N_{i}"]] for i in sim.types}

        if "V" not in constraints:
            if sim.dimension == 3:
                V = extent.TriclinicBox(
                    Lx=thermo[columns["Lx"]],
                    Ly=thermo[columns["Ly"]],
                    Lz=thermo[columns["Lz"]],
                    xy=thermo[columns["xy"]],
                    xz=thermo[columns["xz"]],
                    yz=thermo[columns["yz"]],
                    convention="LAMMPS",
                )
            else:
                V = extent.ObliqueArea(
                    Lx=thermo[columns["Lx"]],
                    Ly=thermo[columns["Ly"]],
                    xy=thermo[columns["xy"]],
                    convention="LAMMPS",
                )
        ens = ensemble.Ensemble(T=T, N=N, V=V, P=P)

        # extract rdfs from trajectory
        if "_rdf_params" in sim[self]:
            sim[self]["_rdf_dump"].process(sim, sim_op)
            rdf = collections.PairMatrix(sim.types)
            num_rdf_samples = 0
            if mpi.world.rank_is_root:
                _rdf_counts = {}
                _rdf_density = {}
                _rdf_num_origins = {}
                for i in sim.types:
                    _rdf_density[i] = 0
                    _rdf_num_origins[i] = 0
                    for j in sim.types:
                        _rdf_counts[i, j] = numpy.zeros(
                            sim[self]["_rdf_params"]["bins"], dtype=int
                        )
                traj = lammpsio.DumpFile(sim[self]["_rdf_file"])
                for snap in traj:
                    # get freud box
                    L = snap.box.high - snap.box.low
                    tilt = snap.box.tilt
                    if tilt is not None:
                        # scale tilt factors to HOOMD convention
                        tilt[0] /= L[1]
                        if sim.dimension == 3:
                            tilt[1] /= L[2]
                            tilt[2] /= L[2]
                    else:
                        tilt = numpy.zeros(3)
                    box_array = numpy.array(
                        [L[0], L[1], L[2], tilt[0], tilt[1], tilt[2]]
                    )
                    if sim.dimension == 2:
                        box_array[2] = 0.0
                        box_array[-2:] = 0.0
                    box = freud.box.Box.from_box(box_array, dimensions=sim.dimension)

                    # determine type masks for RDF calculations
                    # number of particles of each type was already determined above
                    type_masks = {}
                    for i in sim.types:
                        type_masks[i] = snap.typeid == sim["engine"]["types"][i]
                    # build aabb of all particles and generate a parent
                    # neighbor list with the RDF cutoff
                    aabb = freud.locality.AABBQuery(box, snap.position)
                    neighbors = aabb.query(
                        snap.position,
                        dict(
                            mode="ball",
                            r_max=sim[self]["_rdf_params"]["stop"],
                            exclude_ii=True,
                        ),
                    ).toNeighborList()
                    for i in sim.types:
                        _rdf_density[i] += N[i] / box.volume
                        _rdf_num_origins[i] += N[i]
                        for j in sim.types:
                            filter_ij = numpy.logical_and(
                                type_masks[i][neighbors[:, 0]],
                                type_masks[j][neighbors[:, 1]],
                            )
                            counts, _ = numpy.histogram(
                                neighbors.distances[filter_ij],
                                bins=sim[self]["_rdf_params"]["bins"],
                                range=(0, sim[self]["_rdf_params"]["stop"]),
                            )
                            _rdf_counts[i, j] += counts
                    # then do rdfs using the AABBs

                    num_rdf_samples += 1

                for i, j in rdf:
                    if num_rdf_samples > 0:
                        bin_edges = numpy.linspace(
                            0,
                            sim[self]["_rdf_params"]["stop"],
                            sim[self]["_rdf_params"]["bins"] + 1,
                        )
                        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                    if sim.dimension == 3:
                        bin_extents = (4 * numpy.pi / 3) * (
                            bin_edges[1:] ** 3 - bin_edges[:-1] ** 3
                        )
                    elif sim.dimension == 2:
                        bin_extents = numpy.pi * (
                            bin_edges[1:] ** 2 - bin_edges[:-1] ** 2
                        )
                    density = {k: _rdf_density[k] / num_rdf_samples for k in sim.types}
                    g = numpy.zeros_like(bin_centers)
                    if i == j:
                        if _rdf_num_origins[i] > 0 and density[i] > 0:
                            g = _rdf_counts[i, i] / (
                                _rdf_num_origins[i] * density[i] * bin_extents
                            )
                    else:
                        # this takes the weighted average of g_ij and g_ji
                        num_ij_origins = _rdf_num_origins[i] + _rdf_num_origins[j]
                        if num_ij_origins > 0:
                            if density[j] > 0:
                                g += _rdf_counts[i, j] / (
                                    num_ij_origins * density[j] * bin_extents
                                )
                            if density[i] > 0:
                                g += _rdf_counts[j, i] / (
                                    num_ij_origins * density[i] * bin_extents
                                )
                    rdf[i, j] = numpy.column_stack((bin_centers, g))
            # sync across ranks and convert to RDF object
            num_rdf_samples = mpi.world.bcast(num_rdf_samples)
            for pair in rdf:
                gr = mpi.world.bcast_numpy(rdf[pair])
                ens.rdf[pair] = ensemble.RDF(gr[:, 0], gr[:, 1])

            del sim[self]["_rdf_dump"]
        else:
            num_rdf_samples = 0

        sim[self]["ensemble"] = ens
        sim[self]["num_thermo_samples"] = None
        sim[self]["num_rdf_samples"] = num_rdf_samples

        # optionally save file
        if self.filename is not None:
            if mpi.world.rank_is_root:
                ens.save(sim.directory.file(self.filename))
            mpi.world.barrier()

    _get_rdf_params = analyze.EnsembleAverage._get_rdf_params

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

        # get T now
        if "T" in constraints:
            thermostat = constraints["T"]
            if thermostat.anneal:
                constraints["T"] = 0.5 * (thermostat.T[0] + thermostat.T[1])
            else:
                constraints["T"] = thermostat.T

        # defer N & V calculations until later...

        return constraints


class Record(AnalysisOperation):
    def __init__(self, filename, every, quantities):
        self.filename = filename
        self.every = every
        self.quantities = quantities

    def _pre_run_commands(self, sim, sim_op):
        # translate quantities into lammps variables
        quantity_map = {
            "potential_energy": "pe",
            "kinetic_energy": "ke",
            "temperature": "temp",
            "pressure": "press",
        }
        var_ids = {}
        cmds = []
        for q in self.quantities:
            var_ids[q] = Counters.new_variable_id()
            cmds.append("variable {} equal {}".format(var_ids[q], quantity_map[q]))

        # write quantities to file with fix ave/time
        if mpi.world.rank == 0:
            file_ = sim.directory.temporary_file()
        else:
            file_ = None
        sim[self]["_fix_id"] = Counters.new_fix_id()
        sim[self]["_log_file"] = mpi.world.bcast(file_)
        cmds.append(
            (
                "fix {fixid} all ave/time {every} 1 {every} {vars}"
                ' mode scalar ave one file {filename} format " %.18e"'
            ).format(
                fixid=sim[self]["_fix_id"],
                every=self.every,
                filename=sim[self]["_log_file"],
                vars=" ".join(["v_" + str(var_ids[q]) for q in self.quantities]),
            )
        )

        # stash variable ids to delete them later
        sim[self]["_var_ids"] = var_ids
        return cmds

    def _post_run_commands(self, sim, sim_op):
        cmds = ["unfix {}".format(sim[self]["_fix_id"])]
        del sim[self]["_fix_id"]
        # delete variables
        for var_id in sim[self]["_var_ids"].values():
            cmds.append("variable {} delete".format(var_id))
        del sim[self]["_var_ids"]

        return cmds

    def process(self, sim, sim_op):
        data = mpi.world.loadtxt(sim[self]["_log_file"])
        sim[self]["timestep"] = data[:, 0].astype(int)
        for i, q in enumerate(self.quantities, start=1):
            sim[self][q] = data[:, i]

        # optionally save file
        if self.filename is not None:
            if mpi.world.rank_is_root:
                analyze.Record._save(
                    sim.directory.file(self.filename), self.quantities, sim[self]
                )
            mpi.world.barrier()


class WriteTrajectory(AnalysisOperation):
    """Writes a LAMMPS dump file.

    When all options are set to True the file has the following format::

        ITEM: ATOMS id type mass x y z vx vy vz ix iy iz

    where ``id`` is the atom ID, ``x y z`` are positions, ``vx vy vz`` are
    velocities, and ``ix iy iz`` are images.

    """

    def __init__(self, filename, every, format, velocities, images, types, masses):
        self.filename = filename
        self.every = every
        self.format = format
        self.velocities = velocities
        self.images = images
        self.types = types
        self.masses = masses

    def _pre_run_commands(self, sim, sim_op):
        schema = analyze.WriteTrajectory._make_lammps_schema(
            self.velocities, self.images, self.types, self.masses
        )

        # determine number of columns from max index
        num_columns = 0
        for column in schema.values():
            if not isinstance(column, int):
                column = max(column)
            if column > num_columns:
                num_columns = column
        num_columns += 1

        dump_format = [None] * num_columns
        for key, column in schema.items():
            if key == "position":
                for col, name in zip(column, ("x", "y", "z")):
                    dump_format[col] = name
            elif key == "velocity":
                for col, name in zip(column, ("vx", "vy", "vz")):
                    dump_format[col] = name
            elif key == "image":
                for col, name in zip(column, ("ix", "iy", "iz")):
                    dump_format[col] = name
            elif key == "typeid":
                dump_format[column] = "type"
            elif key in ("mass", "id"):
                dump_format[column] = key
            else:
                raise KeyError(f"Unknown LAMMPS schema key {key}")
        if None in dump_format:
            raise ValueError("Noncompact schema keys")
        dump_format = " ".join(dump_format)

        dump_id = Counters.new_dump_id()
        cmds = [
            "dump {} all custom {} {} {}".format(
                dump_id, self.every, sim.directory.file(self.filename), dump_format
            ),
            "dump_modify {} append no pbc yes flush yes".format(dump_id),
        ]

        sim[self]["_dump_id"] = dump_id

        return cmds

    def _post_run_commands(self, sim, sim_op):
        cmds = ["undump {}".format(sim[self]["_dump_id"])]
        del sim[self]["_dump_id"]
        return cmds

    def process(self, sim, sim_op):
        filename = sim.directory.file(self.filename)
        file_format = analyze.WriteTrajectory._detect_format(filename, self.format)
        if file_format == "HOOMD-GSD":
            if mpi.world.rank_is_root:
                gsd_file = sim.directory.temporary_file(".gsd")
                type_map = {v: k for k, v in sim["engine"]["types"].items()}
                with gsd.hoomd.open(gsd_file, _gsd_write_mode) as t:
                    for snap in lammpsio.DumpFile(filename, sort_ids=True):
                        frame = snap.to_hoomd_gsd(type_map)
                        t.append(frame)
                shutil.move(gsd_file, filename)
            mpi.world.barrier()


class LAMMPS(simulate.Simulation):
    """Simulation using LAMMPS.

    A simulation is performed using `LAMMPS <https://docs.lammps.org>`_.
    LAMMPS is a molecular dynamics program that can execute on both CPUs and
    GPUs, as a single process or with MPI parallelism. The launch configuration
    will be automatically selected for you when the simulation is run.

    The version of LAMMPS must be 29 Sep 2021 or newer. It is recommended to build
    LAMMPS with its `Python interface <https://docs.lammps.org/Python_head.html>`_.
    However, it is possible to run LAMMPS as a binary by specifying ``executable``::

        relentless.simulate.LAMMPS(init, ops, executable="lmp_serial")

    This can be helpful if you do not have a build of LAMMPS with Python support
    enabled; however, it will typically be a bit slower than running LAMMPS via
    Python. To run LAMMPS as an executable with MPI support, you should **not**
    launch ``relentless`` with ``mpirexec``, and instead should include the
    ``mpiexec` command and options in the ``executable``::

        relentless.simulate.LAMMPS(init, ops, executable="mpiexec -n 8 lmp_mpi")

    .. warning::

        LAMMPS requires that tabulated pair potentials do not include an entry for
        :math:`r = 0`. Make sure to set
        :attr:`~relentless.simulate.PairPotentialTabulator.rmin` to a small value
        larger than 0.

    Parameters
    ----------
    initializer : :class:`~relentless.simulate.SimulationOperation`
        Operation that initializes the simulation.
    operations : array_like
        :class:`~relentless.simulate.SimulationOperation` to execute for run.
        Defaults to ``None``, which means nothing is done after initialization.
    quiet : bool
        If ``True``, silence LAMMPS screen output. Setting this to ``False`` can
        be helpful for debugging but would be very noisy in a long production
        simulation.
    types : dict
        Mapping from relentless types to LAMMPS integer types. This mapping may
        be used during initialization (and is required for some operations).
    executable : str
        LAMMPS executable. If specified, LAMMPS will be run as a binary
        application rather than its Python library.

    Raises
    ------
    ImportError
        If the :mod:`lammps` package is not found.

    """

    def __init__(
        self,
        initializer,
        operations=None,
        quiet=True,
        types=None,
        executable=None,
    ):
        # test executable if it is specified
        if executable is not None:
            # disallow launching LAMMPS under MPI. this may be slightly too strict,
            # but we can relax it later if needed.
            if mpi.world.enabled:
                raise RuntimeError(
                    "LAMMPS cannot be run under MPI with an executable."
                    " Put mpiexec in the executable instead."
                )
            # test the executable
            result = subprocess.run(
                executable + " -help", shell=True, capture_output=True, text=True
            )
            if result.returncode != 0:
                raise OSError(
                    "LAMMPS executable {} failed to launch.".format(executable)
                )
            self.executable = executable
            lines = result.stdout.splitlines()
            for i, line in enumerate(result.stdout.splitlines()):
                if "Large-scale Atomic/Molecular Massively Parallel Simulator" in line:
                    # these split indexes are hardcoded based on standard help output
                    version_str = line.split("-")[2].strip()
                    # then this coerces the version into the LAMMPS integer format
                    self.version = int(
                        datetime.datetime.strptime(version_str, "%d %b %Y").strftime(
                            "%Y%m%d"
                        )
                    )
                elif line == "Installed packages:":
                    installed_packages = []
                    for line_ in lines[i + 2 :]:
                        if len(line_) == 0:
                            break
                        installed_packages += line_.strip().split()
                    self.packages = tuple(installed_packages)
        else:
            if not _lammps_found:
                raise ImportError("LAMMPS not found.")

            self.executable = None
            lmp = lammps.lammps(
                cmdargs=[
                    "-echo",
                    "none",
                    "-log",
                    "none",
                    "-screen",
                    "none",
                    "-nocite",
                ]
            )
            self.version = lmp.version()
            self.packages = tuple(lmp.installed_packages)
            del lmp
        if self.version < 20210929:
            raise ImportError("Only LAMMPS 29 Sep 2021 or newer is supported.")

        super().__init__(initializer, operations)
        self.quiet = quiet
        self.types = types

    def _post_run(self, sim):
        # force all the lammps commands to execute, since the operations did
        # not actually do it
        if not sim["engine"]["use_python"]:
            # send the commands to file
            if mpi.world.rank_is_root:
                file_ = sim.directory.temporary_file()
                with open(file_, "w") as f:
                    for cmd in sim["engine"]["_lammps_commands"]:
                        f.write(cmd + "\n")
            else:
                file_ = None
            file_ = mpi.world.bcast(file_)

            # then run lammps as an executable
            run_cmd = sim["engine"]["_lammps"] + ["-i", file_]
            subprocess.run(" ".join(run_cmd), shell=True, check=True)

        # then keep going
        super()._post_run(sim)

    def _initialize_engine(self, sim):
        if self.quiet:
            launch_args = [
                "-echo",
                "none",
                "-log",
                "none",
                "-screen",
                "none",
                "-nocite",
            ]
        else:
            launch_args = [
                "-echo",
                "screen",
                "-log",
                sim.directory.file("log.lammps"),
                "-nocite",
            ]

        sim["engine"]["version"] = self.version
        sim["engine"]["packages"] = self.packages
        if self.executable is not None:
            sim["engine"]["use_python"] = False
            sim["engine"]["_lammps"] = [self.executable] + launch_args
        else:
            sim["engine"]["use_python"] = True
            sim["engine"]["_lammps"] = lammps.lammps(cmdargs=launch_args)
        sim["engine"]["_lammps_commands"] = []

        sim["engine"]["types"] = self.types
        sim["engine"]["units"] = "lj"
        sim["engine"]["atom_style"] = "atomic"

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
