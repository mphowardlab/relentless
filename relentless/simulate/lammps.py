"""
LAMMPS
======

This module implements the :class:`LAMMPS` simulation backend and its specific
operations. It is best to interface with these operations using the frontend in
:mod:`relentless.simulate`.

.. rubric:: Developer notes

To implement your own LAMMPS operation, create an operation that derives from
:class:`LAMMPSOperation` and define the required methods.

.. autoclass:: LAMMPSOperation
    :members:
    :special-members: __call__
.. autoclass:: Initialize
    :members:

"""
import abc

import numpy

from relentless import ensemble
from relentless import mpi
from relentless.extent import TriclinicBox
from relentless.extent import ObliqueArea
from . import simulate

try:
    import lammps
    _lammps_found = True
except ImportError:
    _lammps_found = False

class LAMMPSOperation(simulate.SimulationOperation):
    """LAMMPS simulation operation."""
    _fix_counter = 1

    def __call__(self, sim):
        """Evaluates the LAMMPS simulation operation.

        Each deriving class of :class:`LAMMPSOperation` must implement a
        :meth:`to_commands()` method that returns a list or tuple of LAMMPS
        commands that can be executed by :meth:`lammps.commands_list()`.

        Parameters
        ----------
        sim : :class:`~relentless.simulate.simulate.Simulation`
            The simulation object.

        """
        cmds = self.to_commands(sim)
        sim.lammps.commands_list(cmds)

    @classmethod
    def new_fix_id(cls):
        """Make a unique new fix ID.

        Returns
        -------
        int
            The fix ID.

        """
        idx = int(LAMMPSOperation._fix_counter)
        LAMMPSOperation._fix_counter += 1
        return idx

    @abc.abstractmethod
    def to_commands(self, sim):
        """Create the LAMMPS commands for the simulation operation.

        All deriving classes must implement this method.

        Parameters
        ----------
        sim : :class:`~relentless.simulate.simulate.Simulation`
            The simulation object.

        Returns
        -------
        array_like
            The LAMMPS commands for this operation.

        """
        pass

## initializers
class Initialize(LAMMPSOperation):
    """Initializes a simulation box and pair potentials.

    Parameters
    ----------
    units : str
        The LAMMPS style of units used in the simulation (defaults to ``lj``).
    atom_style : str
        The LAMMPS style of atoms used in a simulation (defaults to ``atomic``).

    """
    def __init__(self, lammps_types=None, units='lj', atom_style='atomic'):
        self.lammps_types = lammps_types
        self.units = units
        self.atom_style = atom_style

    def __call__(self, sim):
        super().__call__(sim)
        sim[self].lammps_types = dict(self.lammps_types)

    def to_commands(self, sim):
        cmds = ['units {style}'.format(style=self.units),
                'boundary p p p',
                'dimension {dim}'.format(dim=sim.dimension),
                'atom_style {style}'.format(style=self.atom_style)]
        cmds += self.initialize(sim)

        return cmds


class InitializeFromFile(Initialize):
    """Initializes a simulation box and pair potentials from a LAMMPS data file.

    Parameters
    ----------
    filename : str
        The file from which to read the system data.

    """
    def __init__(self, filename):
        super().__init__()
        self.filename = filename

    def initialize(self, sim):
        if self.lammps_types is None:
            raise ValueError('lammps_types needs to be manually specified with a file')
        return ['read_data {filename}'.format(filename=self.filename)]

class InitializeRandomly(Initialize):
    """Initializes a randomly generated simulation box and pair potentials.

    Parameters
    ----------
    seed : int
        The seed to randomly initialize the particle locations.

    Raises
    ------
    ValueError
        If the number of particles is not set for all types.

    """
    def __init__(self, seed, N, V, T=None, masses=None):
        super().__init__({i: idx+1 for idx,i in enumerate(N.keys())})

        self.seed = seed
        self.N = dict(N)
        self.V = V
        if not isinstance(self.V, (TriclinicBox, ObliqueArea)):
            raise TypeError('LAMMPS boxes must be derived from TriclinicBox or ObliqueArea')
        self.T = T
        self.masses = masses

    def initialize(self, sim):
        types = tuple(self.N.keys())

        # make box
        if sim.dimension == 3:
            Lx = self.V.a[0]
            Ly = self.V.b[1]
            Lz = self.V.c[2]
            xy = self.V.b[0]
            xz = self.V.c[0]
            yz = self.V.c[1]
            dL = self.V.a + self.V.b + self.V.c
        elif sim.dimension == 2:
            Lx = self.V.a[0]
            Ly = self.V.b[1]
            Lz = 0.2 # LAMMPS wants Lz to be a tiny number
            xy = self.V.b[0]
            xz = 0.0
            yz = 0.0
            dL = numpy.array((self.V.a[0]+self.V.b[0],self.V.a[1]+self.V.b[1],Lz))
        else:
            raise ValueError('LAMMPS only supports 2d and 3d simulations')
        lo = -0.5*dL
        hi = lo + [Lx,Ly,Lz]
        box_size = numpy.array([lo[0],hi[0],lo[1],hi[1],lo[2],hi[2],xy,xz,yz])
        if not numpy.all(numpy.isclose(box_size[-3:],0)):
            cmds = ['region box prism {} {} {} {} {} {} {} {} {}'.format(*box_size)]
        else:
            cmds = ['region box block {} {} {} {} {} {}'.format(*box_size[:-3])]
        cmds += ['create_box {N} box'.format(N=len(types))]

        # use lammps random initialization routines
        for i in types:
            cmds += ['create_atoms {typeid} random {N} {seed} box'.format(typeid=self.lammps_types[i],
                                                                          N=self.N[i],
                                                                          seed=self.seed+self.lammps_types[i]-1),
                     'mass {typeid} {mass}'.format(typeid=self.lammps_types[i],
                                                   mass=self.masses[i] if self.masses is not None else 1.0)]

        if self.T is not None:
            cmds += ['velocity all create {temp} {seed}'.format(temp=self.T, seed=self.seed)]

        return cmds

## integrators
class MinimizeEnergy(LAMMPSOperation):
    """Runs an energy minimization until converged.

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
        self.options = options
        if 'max_evaluations' not in self.options:
            self.options['max_evaluations'] = None

    def to_commands(self, sim):
        if self.options['max_evaluations'] is None:
            self.options['max_evaluations'] = 100*self.max_iterations

        cmds = ['minimize {etol} {ftol} {maxiter} {maxeval}'.format(etol=self.energy_tolerance,
                                                                    ftol=self.force_tolerance,
                                                                    maxiter=self.max_iterations,
                                                                    maxeval=self.options['max_evaluations'])]
        if sim.dimension == 2:
            fix_2d = self.new_fix_id()
            cmds = ['fix {} all enforce2d'.format(fix_2d)] + cmds + ['unfix {}'.format(fix_2d)]

        return cmds

class AddMDIntegrator(LAMMPSOperation):
    """Adds an integrator (for equations of motion) to the simulation.

    Parameters
    ----------
    dt : float
        Time step size for each simulation iteration.

    """
    def __init__(self, dt):
        self.dt = dt

    def to_commands(self, sim):
        return ['timestep {}'.format(self.dt)]

class AddLangevinIntegrator(AddMDIntegrator):
    """Langevin dynamics for a NVE ensemble.

    Parameters
    ----------
    dt : float
        Time step size for each simulation iteration.
    friction : float or dict
        Drag coefficient for each particle type (shared or per-type).
    seed : int
        Seed used to randomly generate a uniform force.

    Raises
    ------
    ValueError
        If particle masses are not set for all types.
    ValueError
        If the friction factor is not set as a single value or per-type
        for all types.

    """
    def __init__(self, dt, T, friction, seed):
        super().__init__(dt)
        self.T = T
        self.friction = friction
        self.seed = seed

        self._fix_langevin = self.new_fix_id()
        self._fix_nve = self.new_fix_id()

    def to_commands(self, sim):
        cmds = super().to_commands(sim)

        # obtain per-type mass (arrays 1-indexed using lammps convention)
        Ntypes = len(sim.types)
        mass = sim.lammps.numpy.extract_atom('mass')
        if mass is None or mass.shape != (Ntypes+1,):
            raise ValueError('Per-type masses not set.')
        mass = numpy.squeeze(mass)

        # obtain per-type friction factor
        friction = numpy.zeros_like(mass)
        for t in sim.types:
            try:
                friction[sim[sim.initializer].lammps_types[t]] = self.friction[t]
            except TypeError:
                friction[sim[sim.initializer].lammps_types[t]] = self.friction
            except KeyError:
                raise KeyError('The friction factor for type {} is not set.'.format(t))

        # compute per-type damping parameter and rescale if multiple types
        damp = numpy.divide(mass, friction, where=(friction>0))
        damp_ref = damp[1]
        if Ntypes>1:
            scale = damp/damp_ref
            scale_str = ' '.join(['scale {} {}'.format(i+1,s) for i,s in enumerate(scale[2:])])
        else:
            scale_str = ''

        cmds += ['fix {idx} {group_idx} nve'.format(idx=self._fix_nve,
                                                   group_idx='all'),
                'fix {idx} {group_idx} langevin {t_start} {t_stop} {damp} {seed} {scaling}'.format(idx=self._fix_langevin,
                                                                                                  group_idx='all',
                                                                                                  t_start=self.T,
                                                                                                  t_stop=self.T,
                                                                                                  damp=damp_ref,
                                                                                                  seed=self.seed,
                                                                                                  scaling=scale_str)
               ]

        return cmds

class RemoveLangevinIntegrator(LAMMPSOperation):
    """Removes the Langevin integrator operation.

    Parameters
    ----------
    add_op : :class:`AddLangevinIntegrator`
        The integrator addition operation to be removed.

    Raises
    ------
    TypeError
        If the specified addition operation is not a Langevin integrator.

    """
    def __init__(self, add_op):
        self.add_op = add_op

    def to_commands(self, sim):
        cmds = ['unfix {idx}'.format(idx=self.add_op._fix_langevin),
                'unfix {idx}'.format(idx=self.add_op._fix_nve)]

        return cmds

class AddVerletIntegrator(AddMDIntegrator):
    """Family of Verlet integration modes.

    This method supports:

    - NVE integration
    - NVT integration with Nosé-Hoover or Berendsen thermostat
    - NPH integration with MTK or Berendsen barostat
    - NPT integration with Nosé-Hoover or Berendsen thermostat and MTK or Berendsen barostat

    Parameters
    ----------
    dt : float
        Time step size for each simulation iteration.
    thermostat : :class:`~relentless.simulate.simulate.Thermostat`
        Thermostat used for integration (defaults to ``None``).
    barostat : :class:`~relentless.simulate.simulate.Barostat`
        Barostat used for integration (defaults to ``None``).

    Raises
    ------
    TypeError
        If an appropriate combination of thermostat and barostat is not set.

    """
    def __init__(self, dt, thermostat=None, barostat=None):
        super().__init__(dt)
        self.thermostat = thermostat
        self.barostat = barostat

        self._fix = self.new_fix_id()
        self._extra_fixes = []

    def to_commands(self, sim):
        cmds = super().to_commands(sim)
        fix_berendsen_temp = False
        fix_berendsen_pres = False

        if ((self.thermostat is None or isinstance(self.thermostat, simulate.BerendsenThermostat)) and
            (self.barostat is None or isinstance(self.barostat, simulate.BerendsenBarostat))):
            cmds += ['fix {idx} {group_idx} nve'.format(idx=self._fix,
                                                       group_idx='all')]
            if isinstance(self.thermostat, simulate.BerendsenThermostat):
                fix_berendsen_temp = True
            if isinstance(self.barostat, simulate.BerendsenBarostat):
                fix_berendsen_pres = True
        elif (isinstance(self.thermostat, simulate.NoseHooverThermostat) and
             (self.barostat is None or isinstance(self.barostat, simulate.BerendsenBarostat))):
            cmds += ['fix {idx} {group_idx} nvt temp {Tstart} {Tstop} {Tdamp}'.format(idx=self._fix,
                                                                                     group_idx='all',
                                                                                     Tstart=self.thermostat.T,
                                                                                     Tstop=self.thermostat.T,
                                                                                     Tdamp=self.thermostat.tau)]
            if isinstance(self.barostat, simulate.BerendsenBarostat):
                fix_berendsen_pres = True
        elif ((self.thermostat is None or isinstance(self.thermostat, simulate.BerendsenThermostat)) and
              isinstance(self.barostat, simulate.MTKBarostat)):
            cmds += ['fix {idx} {group_idx} nph iso {Pstart} {Pstop} {Pdamp}'.format(idx=self._fix,
                                                                                    group_idx='all',
                                                                                    Pstart=self.barostat.P,
                                                                                    Pstop=self.barostat.P,
                                                                                    Pdamp=self.barostat.tau)]
            if isinstance(self.thermostat, simulate.BerendsenThermostat):
                fix_berendsen_temp = True
        elif isinstance(self.thermostat, simulate.NoseHooverThermostat) and isinstance(self.barostat, simulate.MTKBarostat):
            cmds += ['fix {idx} {group_idx} npt temp {Tstart} {Tstop} {Tdamp} iso {Pstart} {Pstop} {Pdamp}'.format(idx=self._fix,
                                                                                                                  group_idx='all',
                                                                                                                  Tstart=self.thermostat.T,
                                                                                                                  Tstop=self.thermostat.T,
                                                                                                                  Tdamp=self.thermostat.tau,
                                                                                                                  Pstart=self.barostat.P,
                                                                                                                  Pstop=self.barostat.P,
                                                                                                                  Pdamp=self.barostat.tau)]
        else:
            raise TypeError('An appropriate combination of thermostat and barostat must be set.')

        if fix_berendsen_temp:
            _fix_berendsen_t = self.new_fix_id()
            self._extra_fixes.append(_fix_berendsen_t)
            cmds += ['fix {idx} {group_idx} temp/berendsen {Tstart} {Tstop} {Tdamp}'.format(idx=_fix_berendsen_t,
                                                                                            group_idx='all',
                                                                                            Tstart=self.thermostat.T,
                                                                                            Tstop=self.thermostat.T,
                                                                                            Tdamp=self.thermostat.tau)]
        if fix_berendsen_pres:
            _fix_berendsen_p = self.new_fix_id()
            self._extra_fixes.append(_fix_berendsen_p)
            cmds += ['fix {idx} {group_idx} press/berendsen iso {Pstart} {Pstop} {Pdamp}'.format(idx=_fix_berendsen_p,
                                                                                                 group_idx='all',
                                                                                                 Pstart=self.barostat.P,
                                                                                                 Pstop=self.barostat.P,
                                                                                                 Pdamp=self.barostat.tau)]

        return cmds

class RemoveVerletIntegrator(LAMMPSOperation):
    """Removes the Verlet integrator operation.

    Parameters
    ----------
    add_op : :class:`AddVerletIntegrator`
        The integrator addition operation to be removed.

    Raises
    ------
    TypeError
        If the specified addition operation is not a Verlet integrator.

    """
    def __init__(self, add_op):
        self.add_op = add_op

    def to_commands(self, sim):
        cmds = ['unfix {idx}'.format(idx=self.add_op._fix)]
        for _extra_fix in self.add_op._extra_fixes:
            cmds += ['unfix {idx}'.format(idx=_extra_fix)]

        return cmds

class Run(LAMMPSOperation):
    """Advances the simulation by a given number of time steps.

    Parameters
    ----------
    steps : int
        Number of steps to run.

    """
    def __init__(self, steps):
        self.steps = steps

    def to_commands(self, sim):
        cmds = ['run {N}'.format(N=self.steps)]
        if sim.dimension == 2:
            fix_2d = self.new_fix_id()
            cmds = ['fix {} all enforce2d'.format(fix_2d)] + cmds + ['unfix {}'.format(fix_2d)]

        return cmds

class RunUpTo(LAMMPSOperation):
    """Advances the simulation up to a given time step number.

    Parameters
    ----------
    step : int
        Step number up to which to run.

    """
    def __init__(self, step):
        self.step = step

    def to_commands(self, sim):
        cmds = ['run {N} upto'.format(N=self.step)]
        if sim.dimension == 2:
            fix_2d = self.new_fix_id()
            cmds = ['fix {} all enforce2d'.format(fix_2d)] + cmds + ['unfix {}'.format(fix_2d)]

        return cmds

## analyzers
class AddEnsembleAnalyzer(LAMMPSOperation):
    """Analyzes the simulation ensemble and rdf at specified timestep intervals.

    Parameters
    ----------
    check_thermo_every : int
        Interval of time steps at which to log thermodynamic properties of the simulation.
    check_rdf_every : int
        Interval of time steps at which to log the rdf of the simulation.
    rdf_dr : float
        The width (in units ``r``) of a bin in the histogram of the rdf.

    Raises
    ------
    RuntimeError
        If more than one LAMMPS :class:`AddEnsembleAnalyzer` is initialized
        at the same time.

    """
    def __init__(self, check_thermo_every, check_rdf_every, rdf_dr):
        self.check_thermo_every = check_thermo_every
        self.check_rdf_every = check_rdf_every
        self.rdf_dr = rdf_dr

    def to_commands(self, sim):
        # check that IDs reserved for analysis do not yet exist
        reserved_ids = [('fix','thermo_avg'),
                        ('compute','rdf'),
                        ('fix','rdf_avg'),
                        ('variable','T'),
                        ('variable','P'),
                        ('variable','Lx'),
                        ('variable','Ly'),
                        ('variable','Lz'),
                        ('variable','xy'),
                        ('variable','xz'),
                        ('variable','yz')]
        for i in sim.types:
            typeid = sim[sim.initializer].lammps_types[i]
            reserved_ids.append(['group', 'type_{typeid}'.format(typeid=typeid)])
            reserved_ids.append(['variable', 'N_{typeid}'.format(typeid=typeid)])
        for category,name in reserved_ids:
            if sim.lammps.has_id(category,'ensemble_'+name):
                raise RuntimeError('Only one AddEnsembleAnalyzer operation can be used with LAMMPS.')

        # thermodynamic properties
        sim[self].thermo_file = sim.directory.file('lammps_thermo.dat')
        cmds = ['thermo {every}'.format(every=self.check_thermo_every),
                'thermo_style custom temp press lx ly lz xy xz yz',
                'thermo_modify norm no flush no',
                'variable ensemble_T equal temp',
                'variable ensemble_P equal press',
                'variable ensemble_Lx equal lx',
                'variable ensemble_Ly equal ly',
                'variable ensemble_Lz equal lz',
                'variable ensemble_xy equal xy',
                'variable ensemble_xz equal xz',
                'variable ensemble_yz equal yz']
        N_vars = []
        for i in sim.types:
            typeid = sim[sim.initializer].lammps_types[i]
            cmds += ['group ensemble_type_{typeid} type {typeid}'.format(typeid=typeid),
                     'variable ensemble_N_{typeid} equal "count(ensemble_type_{typeid})"'.format(typeid=typeid)]
            N_vars.append('v_ensemble_N_{typeid}'.format(typeid=typeid))
        cmds += [
                ('fix ensemble_thermo_avg all ave/time {every} 1 {every}'
                 ' v_ensemble_T v_ensemble_P'
                 ' v_ensemble_Lx v_ensemble_Ly v_ensemble_Lz'
                 ' v_ensemble_xy v_ensemble_xz v_ensemble_yz'
                 + ' ' + ' '.join(N_vars) +
                 ' mode scalar ave running'
                 ' file {filename} overwrite format " %.16e"').format(every=self.check_thermo_every,
                                                                       filename=sim[self].thermo_file)
                ]

        # pair distribution function
        rmax = sim.potentials.pair.r[-1]
        sim[self].num_bins = numpy.round(rmax/self.rdf_dr).astype(int)
        sim[self].rdf_file = sim.directory.file('lammps_rdf.dat')
        sim[self].rdf_pairs = tuple(sim.pairs)
        # string format lammps arguments based on pairs
        # _pairs is the list of all pairs by LAMMPS type id, in ensemble order
        # _computes is the RDF values for each pair, with the r bin centers prepended
        _pairs = []
        _computes = ['c_ensemble_rdf[1]']
        for idx,(i,j) in enumerate(sim[self].rdf_pairs):
            _pairs.append('{} {}'.format(sim[sim.initializer].lammps_types[i],sim[sim.initializer].lammps_types[j]))
            _computes.append('c_ensemble_rdf[{}]'.format(2*(idx+1)))
        cmds += ['compute ensemble_rdf all rdf {bins} {pairs}'.format(bins=sim[self].num_bins,pairs=' '.join(_pairs)),
                 ('fix ensemble_rdf_avg all ave/time {every} 1 {every}'
                  ' {computes} mode vector ave running off 1'
                  ' file {filename} overwrite format " %.16e"').format(every=self.check_rdf_every,
                                                                         computes=' '.join(_computes),
                                                                         filename=sim[self].rdf_file)
                ]

        return cmds

    def extract_ensemble(self, sim):
        """Creates an ensemble with the averaged thermodynamic properties and rdf.

        Parameters
        ----------
        sim : :class:`~relentless.simulate.simulate.Simulation`
            The simulation object.

        Returns
        -------
        :class:`~relentless.ensemble.Ensemble`
            Ensemble with averaged thermodynamic properties and rdf.

        """
        # extract thermo properties
        # we skip the first 2 rows, which are LAMMPS junk, and slice out the timestep from col. 0
        thermo = mpi.world.loadtxt(sim[self].thermo_file,skiprows=2)[1:]

        num_types = len(sim.types)
        N = {i: Ni for i,Ni in zip(sim.types, thermo[8:8+num_types])}

        V = TriclinicBox(Lx=thermo[2],Ly=thermo[3],Lz=thermo[4],
                         xy=thermo[5],xz=thermo[6],yz=thermo[7],
                         convention=TriclinicBox.Convention.LAMMPS)
        ens = ensemble.Ensemble(N=N,
                                T=thermo[0],
                                P=thermo[1],
                                V=V)

        # extract rdfs
        # LAMMPS injects a column for the row index, so we start at column 1 for r
        # we skip the first 4 rows, which are LAMMPS junk, and slice out the first column
        rdf = mpi.world.loadtxt(sim[self].rdf_file,skiprows=4)[:,1:]
        for i,pair in enumerate(sim[self].rdf_pairs):
            ens.rdf[pair] = ensemble.RDF(rdf[:,0],rdf[:,i+1])

        return ens

class LAMMPS(simulate.Simulation):
    """Simulation using LAMMPS.

    A simulation is performed using `LAMMPS <https://docs.lammps.org>`_.
    LAMMPS is a molecular dynamics program that can execute on both CPUs and
    GPUs, as a single process or with MPI parallelism. The launch configuration
    will be automatically selected for you when the simulation is run.

    LAMMPS must be built with its `Python interface <https://docs.lammps.org/Python_head.html>`_
    and must be version 29 Sep 2021 or newer.

    Raises
    ------
    ImportError
        If the :mod:`lammps` package is not found.

    """
    def __init__(self, initializer, operations=None, dimension=3, quiet=True):
        if not _lammps_found:
            raise ImportError('LAMMPS not found.')

        super().__init__(initializer, operations)
        self.dimension = dimension
        self.quiet = quiet

    def _new_instance(self, initializer, potentials, directory):
        sim = super()._new_instance(initializer, potentials, directory)

        # add the lammps engine to the instance
        if self.quiet:
            # create lammps instance with all output disabled
            launch_args = ['-echo','none',
                           '-log','none',
                           '-screen','none',
                           '-nocite']
        else:
            launch_args = ['-echo','screen',
                           '-log', sim.directory.file('log.lammps'),
                           '-nocite']
        sim.lammps = lammps.lammps(cmdargs=launch_args)
        if sim.lammps.version() < 20210929:
            raise ImportError('Only LAMMPS 29 Sep 2021 or newer is supported.')

        # run the initializer
        sim.dimension = self.dimension
        initializer(sim)
        sim.types = sim[initializer].lammps_types.keys()

        # attach the potentials
        self._attach_potentials(sim)

        return sim

    def _attach_potentials(self, sim):
        """Adds tabulated pair potentials to the simulation object.

        Parameters
        ----------
        sim: :class:`~relentless.simulate.simulate.Simulation`
            The simulation object.

        Returns
        -------
        array_like
            The LAMMPS commands for attaching pair potentials.

        Raises
        ------
        ValueError
            If there are not at least two points in the tabulated potential.
        ValueError
            If the pair potentials do not have equally spaced ``r``.

        """
        # lammps requires r > 0
        flags = sim.potentials.pair.r > 0
        r = sim.potentials.pair.r[flags]
        Nr = len(r)
        if Nr == 1:
            raise ValueError('LAMMPS requires at least two points in the tabulated potential.')

        # check that all r are equally spaced
        dr = r[1:]-r[:-1]
        if not numpy.all(numpy.isclose(dr,dr[0])):
            raise ValueError('LAMMPS requires equally spaced r in pair potentials.')

        def pair_map(sim,pair):
            # Map lammps type indexes as a pair, lowest type first
            i,j = pair
            id_i = sim[sim.initializer].lammps_types[i]
            id_j = sim[sim.initializer].lammps_types[j]
            if id_i > id_j:
                id_i,id_j = id_j,id_i

            return id_i,id_j

        # write all potentials into a file
        file_ = sim.directory.file('lammps_pair_table.dat')
        if mpi.world.rank_is_root:
            with open(file_,'w') as fw:
                fw.write('# LAMMPS tabulated pair potentials\n')
                rcut = {}
                for i,j in sim.pairs:
                    id_i,id_j = pair_map(sim,(i,j))
                    fw.write(('# pair ({i},{j})\n'
                              '\n'
                              'TABLE_{id_i}_{id_j}\n').format(i=i,
                                                              j=j,
                                                              id_i=id_i,
                                                              id_j=id_j)
                            )
                    fw.write('N {N} R {rmin} {rmax}\n\n'.format(N=Nr,
                                                                rmin=r[0],
                                                                rmax=r[-1]))

                    u = sim.potentials.pair.energy((i,j))[flags]
                    f = sim.potentials.pair.force((i,j))[flags]
                    for idx in range(Nr):
                        fw.write('{idx} {r} {u} {f}\n'.format(idx=idx+1,r=r[idx],u=u[idx],f=f[idx]))

                    # find r where potential and force are zero
                    nonzero_r = numpy.flatnonzero(numpy.logical_and(~numpy.isclose(u,0),~numpy.isclose(f,0)))
                    if len(nonzero_r) > 1:
                        # cutoff at last nonzero r (cannot be first r)
                        rcut[(i,j)] = r[nonzero_r[-1]]
                    else:
                        # if first or second r is nonzero, cutoff at second
                        rcut[(i,j)] = r[1]
        else:
            rcut = None
        rcut = mpi.world.bcast(rcut)

        # process all lammps commands
        cmds = ['neighbor {skin} multi'.format(skin=sim.potentials.pair.neighbor_buffer)]
        cmds += ['pair_style table linear {N}'.format(N=Nr)]

        for i,j in sim.pairs:
            # get lammps type indexes, lowest type first
            id_i,id_j = pair_map(sim,(i,j))
            cmds += ['pair_coeff {id_i} {id_j} {filename} TABLE_{id_i}_{id_j} {cutoff}'.format(id_i=id_i,
                                                                                               id_j=id_j,
                                                                                               filename=file_,
                                                                                               cutoff=rcut[(i,j)])]

        sim.lammps.commands_list(cmds)

    # initialization
    InitializeFromFile = InitializeFromFile
    InitializeRandomly = InitializeRandomly

    # energy minimization
    MinimizeEnergy = MinimizeEnergy

    # md integrators
    AddLangevinIntegrator = AddLangevinIntegrator
    RemoveLangevinIntegrator = RemoveLangevinIntegrator
    AddVerletIntegrator = AddVerletIntegrator
    RemoveVerletIntegrator = RemoveVerletIntegrator

    # run commands
    Run = Run
    RunUpTo = RunUpTo

    # analysis
    AddEnsembleAnalyzer = AddEnsembleAnalyzer
