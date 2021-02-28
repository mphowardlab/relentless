import abc
import os

import numpy as np

from relentless.ensemble import RDF
from relentless.volume import TriclinicBox
from . import simulate

try:
    import lammps
    _lammps_found = True
except ImportError:
    _lammps_found = False

class LAMMPS(simulate.Simulation):
    """:py:class:`Simulation` using LAMMPS framework.

    LAMMPS must be built with its `Python interface <https://lammps.sandia.gov/doc/Python_head.html>`_
    and must be version 29 Oct 2020 or newer.

    Raises
    ------
    ImportError
        If the `lammps` package is not found.

    """
    def __init__(self, operations=None, quiet=True, **options):
        if not _lammps_found:
            raise ImportError('LAMMPS not found.')

        super().__init__(operations,**options)
        self.quiet = quiet

    def _new_instance(self, ensemble, potentials, directory):
        sim = super()._new_instance(ensemble,potentials,directory)

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
        if sim.lammps.version() < 20201029:
            raise ImportError('Only LAMMPS 29 Oct 2020 or newer is supported.')

        # lammps uses 1-indexed ints for types, so build mapping in both direction
        sim.type_map = {}
        sim.typeid_map = {}
        for i,t in enumerate(sim.ensemble.types):
            sim.type_map[t] = i+1
            sim.typeid_map[sim.type_map[t]] = t

        return sim

class LAMMPSOperation(simulate.SimulationOperation):
    _fix_counter = 1

    def __call__(self, sim):
        """Evaluates the LAMMPS simulation operation.

        Each deriving class of :py:class:`LAMMPSOperation` must implement a
        :py:meth:`to_commands()` method that returns a list or tuple of LAMMPS
        commands that can be executed by :py:meth:`lammps.commands_list()`.

        Parameters
        ----------
        sim : :py:class:`Simulation`
            The simulation object.

        """
        cmds = self.to_commands(sim)
        sim.lammps.commands_list(cmds)

    @classmethod
    def new_fix_id(cls):
        """Sets a unique new ID for a LAMMPS fix.

        Returns
        -------
        int
            The fix ID.

        """
        idx = int(cls._fix_counter)
        cls._fix_counter += 1
        return idx

    @abc.abstractmethod
    def to_commands(self, sim):
        """Calls the appropriate LAMMPS commands for the simulation operation.

        All classes deriving from :py:class:`LAMMPSOperation` must implement
        this method.

        Parameters
        ----------
        sim : :py:class:`Simulation`
            The simulation object.

        Returns
        -------
        array_like
            The LAMMPS commands for this operation.

        """
        pass

class Initialize(LAMMPSOperation):
    """Initializes a simulation box and pair potentials.

    Parameters
    ----------
    neighbor_buffer : float
        Buffer width.
    units : str
        The LAMMPS style of units used in the simulation (defaults to `lj`).
    atom_style : str
        The LAMMPS style of atoms used in a simulation (defaults to `atomic`).

    """
    def __init__(self, neighbor_buffer, units='lj', atom_style='atomic'):
        self.neighbor_buffer = neighbor_buffer
        self.units = units
        self.atom_style = atom_style

    def to_commands(self, sim):
        cmds = ['units {style}'.format(style=self.units),
                'boundary p p p',
                'atom_style {style}'.format(style=self.atom_style)]

        return cmds

    def extract_box_params(self, sim):
        """Extracts LAMMPS box parameters (*Lx*, *Ly*, *Lz*, *xy*, *xz*, *yz*)
        from the simulation's ensemble volume.

        Parameters
        ----------
        sim: :py:class:`Simulation`
            The simulation object.

        Returns
        -------
        array_like
            Array of the simulation box parameters.

        Raises
        ------
        ValueError
            If the volume is not set.
        TypeError
            If the volume does not derive from :py:class:`TriclinicBox`.

        """
        # cast simulation box in LAMMPS parameters
        V = sim.ensemble.V
        if V is None:
            raise ValueError('Box volume must be set.')
        elif not isinstance(V, TriclinicBox):
            raise TypeError('LAMMPS boxes must be derived from TriclinicBox')

        Lx = V.a[0]
        Ly = V.b[1]
        Lz = V.c[2]
        xy = V.b[0]
        xz = V.c[0]
        yz = V.c[1]

        lo = -0.5*np.array([Lx,Ly,Lz])
        hi = lo + V.a + V.b + V.c

        return np.array([lo[0],hi[0],lo[1],hi[1],lo[2],hi[2],xy,xz,yz])

    def attach_potentials(self, sim):
        """Adds tabulated pair potentials to the simulation object.

        Parameters
        ----------
        sim: :py:class:`Simulation`
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
            If the pair potentials do not have equally spaced r.

        """
        # lammps requires r > 0
        flags = sim.potentials.pair.r > 0
        r = sim.potentials.pair.r[flags]
        Nr = len(r)
        if Nr == 1:
            raise ValueError('LAMMPS requires at least two points in the tabulated potential.')

        # check that all r are equally spaced
        dr = r[1:]-r[:-1]
        if not np.all(np.isclose(dr,dr[0])):
            raise ValueError('LAMMPS requires equally spaced r in pair potentials.')

        def pair_map(sim,pair):
            # Map lammps type indexes as a pair, lowest type first
            i,j = pair
            id_i = sim.type_map[i]
            id_j = sim.type_map[j]
            if id_i > id_j:
                id_i,id_j = id_j,id_i

            return id_i,id_j

        # write all potentials into a file
        file_ = sim.directory.file('lammps_pair_table.dat')
        with open(file_,'w') as fw:
            fw.write('# LAMMPS tabulated pair potentials\n')
            for i,j in sim.ensemble.pairs:
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

        # process all lammps commands
        cmds = ['neighbor {skin} multi'.format(skin=self.neighbor_buffer)]
        cmds += ['pair_style table linear {N}'.format(N=Nr)]
        for i,j in sim.ensemble.pairs:
            # get lammps type indexes, lowest type first
            id_i,id_j = pair_map(sim,(i,j))
            cmds += ['pair_coeff {id_i} {id_j} {filename} TABLE_{id_i}_{id_j}'.format(id_i=id_i,id_j=id_j,filename=file_)]

        return cmds

class InitializeFromFile(Initialize):
    """Initializes a simulation box and pair potentials from a LAMMPS data file.

    Parameters
    ----------
    filename : str
        The file from which to read the system data.
    neighbor_buffer : float
        Buffer width.
    units : str
        The LAMMPS style of units used in the simulation (defaults to `lj`).
    atom_style : str
        The LAMMPS style of atoms used in a simulation (defaults to `atomic`).

    """
    def __init__(self, filename, neighbor_buffer, units='lj', atom_style='atomic'):
        super().__init__(neighbor_buffer, units, atom_style)
        self.filename = filename

    def to_commands(self, sim):
        cmds = super().to_commands(sim)
        cmds += ['read_data {filename}'.format(filename=self.filename)]
        cmds += self.attach_potentials(sim)

        return cmds

class InitializeRandomly(Initialize):
    """Initializes a randomly generated simulation box and pair potentials.

    The number of particles must be set for all types.

    Parameters
    ----------
    seed : int
        The seed to randomly initialize the particle locations.
    neighbor_buffer : float
        Buffer width.
    units : str
        The LAMMPS style of units used in the simulation (defaults to `lj`).
    atom_style : str
        The LAMMPS style of atoms used in a simulation (defaults to `atomic`).

    """
    def __init__(self, seed, neighbor_buffer, units='lj', atom_style='atomic'):
        super().__init__(neighbor_buffer, units, atom_style)
        self.seed = seed

    def to_commands(self, sim):
        cmds = super().to_commands(sim)

        # make box from ensemble
        box = self.extract_box_params(sim)
        if not np.all(np.isclose(box[-3:],0)):
            cmds += ['region box prism {} {} {} {} {} {} {} {} {}'.format(*box)]
        else:
            cmds += ['region box block {} {} {} {} {} {}'.format(*box[:-3])]
        cmds += ['create_box {N} box'.format(N=len(sim.ensemble.types))]

        # use lammps random initialization routines
        for i in sim.ensemble.types:
            if sim.ensemble.N[i] is None:
                raise ValueError('Number of particles for type {} must be set.'.format(i))

            cmds += ['create_atoms {typeid} random {N} {seed} box'.format(typeid=sim.type_map[i],
                                                                          N=sim.ensemble.N[i],
                                                                          seed=self.seed+sim.type_map[i]-1)]
        cmds += ['mass * 1.0',
                 'velocity all create {temp} {seed}'.format(temp=sim.ensemble.T,
                                                            seed=self.seed)]

        cmds += self.attach_potentials(sim)

        return cmds

class MinimizeEnergy(LAMMPSOperation):
    """Runs an energy minimization until converged.

    Parameters
    ----------
    energy_tolerance : float
        Energy convergence criterion.
    force_tolerance : float
        Force convergence criterion.
    max_iterations : int
        Maximum number of iterations to run the minimization.
    dt : float
        Maximum step size.

    """
    def __init__(self, energy_tolerance, force_tolerance, max_iterations, dt):
        self.energy_tolerance = energy_tolerance
        self.force_tolerance = force_tolerance
        self.max_iterations = max_iterations

    def to_commands(self, sim):
        cmds = ['minimize {etol} {ftol} {maxiter} {maxeval}'.format(etol=self.energy_tolerance,
                                                                    ftol=self.force_tolerance,
                                                                    maxiter=self.max_iterations,
                                                                    maxeval=100*self.max_iterations)]

        return cmds

class AddLangevinIntegrator(LAMMPSOperation):
    """Langevin dynamics for a NVE ensemble.

    The simulation ensemble must be NVT, with all per-type particle masses set.
    The friction factor can be a single value or defined per-type for all types.

    Parameters
    ----------
    dt : float
        Time step size for each simulation iteration.
    friction : float or dict
        Drag coefficient for each particle type (shared or per-type).
    seed : int
        Seed used to randomly generate a uniform force.

    """
    def __init__(self, dt, friction, seed):
        self.friction = friction
        self.seed = seed

        self._fix_langevin = self.new_fix_id()
        self._fix_nve = self.new_fix_id()

    def to_commands(self, sim):
        if not sim.ensemble.aka('NVT'):
            raise ValueError('Simulation ensemble is not NVT.')

        # obtain per-type mass (arrays 1-indexed using lammps convention)
        Ntypes = len(sim.ensemble.types)
        mass = sim.lammps.numpy.extract_atom('mass')
        if mass is None or mass.shape != (Ntypes+1,1):
            raise ValueError('Per-type masses not set.')
        mass = np.squeeze(mass)

        # obtain per-type friction factor
        friction = np.zeros_like(mass)
        for t in sim.ensemble.types:
            try:
                friction[sim.type_map[t]] = self.friction[t]
            except TypeError:
                friction[sim.type_map[t]] = self.friction
            except KeyError:
                raise KeyError('The friction factor for type {} is not set.'.format(t))

        # compute per-type damping parameter and rescale if multiple types
        damp = np.divide(mass, friction, where=(friction>0))
        damp_ref = damp[1]
        if Ntypes>1:
            scale = damp/damp_ref
            scale_str = ' '.join(['scale {} {}'.format(i+1,s) for i,s in enumerate(scale[2:])])
        else:
            scale_str = ''

        cmds = ['fix {idx} {group_idx} nve'.format(idx=self._fix_nve,
                                                   group_idx='all'),
                'fix {idx} {group_idx} langevin {t_start} {t_stop} {damp} {seed} {scaling}'.format(idx=self._fix_langevin,
                                                                                                  group_idx='all',
                                                                                                  t_start=sim.ensemble.T,
                                                                                                  t_stop=sim.ensemble.T,
                                                                                                  damp=damp_ref,
                                                                                                  seed=self.seed,
                                                                                                  scaling=scale_str)
               ]

        return cmds

class RemoveLangevinIntegrator(LAMMPSOperation):
    """Removes the Langevin integrator operation.

    Parameters
    ----------
    add_op : :py:class:`AddLangevinIntegrator`
        The integrator addition operation to be removed.

    Raises
    ------
    TypeError
        If the specified addition operation is not a Langevin integrator.

    """
    def __init__(self, add_op):
        if not isinstance(add_op, AddLangevinIntegrator):
            raise TypeError('Addition operation is not AddLangevinIntegrator.')
        self.add_op = add_op

    def to_commands(self, sim):
        cmds = ['unfix {idx}'.format(idx=self.add_op._fix_langevin),
                'unfix {idx}'.format(idx=self.add_op._fix_nve)]

        return cmds

class AddNPTIntegrator(LAMMPSOperation):
    r"""NPT integration via Nos\'e-Hoover thermostat/barostat.

    The simulation ensemble must be NPT.

    Parameters
    ----------
    dt : float
        Time step size for each simulation iteration.
    tau_T : float
        Coupling constant for the thermostat.
    tau_P : float
        Coupling constant for the barostat.

    """
    def __init__(self, dt, tau_T, tau_P):
        self.tau_T = tau_T
        self.tau_P = tau_P

        self._fix = super().new_fix_id()

    def to_commands(self, sim):
        if not sim.ensemble.aka('NPT'):
            raise ValueError('Simulation ensemble is not NPT.')
        cmds = ['fix {idx} {group_idx} npt temp {Tstart} {Tstop} {Tdamp} iso {Pstart} {Pstop} {Pdamp}'.format(idx=self._fix,
                                                                                                              group_idx='all',
                                                                                                              Tstart=sim.ensemble.T,
                                                                                                              Tstop=sim.ensemble.T,
                                                                                                              Tdamp=self.tau_T,
                                                                                                              Pstart=sim.ensemble.P,
                                                                                                              Pstop=sim.ensemble.P,
                                                                                                              Pdamp=self.tau_P)]

        return cmds

class RemoveNPTIntegrator(LAMMPSOperation):
    """Removes the NPT integrator operation.

    Parameters
    ----------
    add_op : :py:class:`AddNPTIntegrator`
        The integrator addition operation to be removed.

    Raises
    ------
    TypeError
        If the specified addition operation is not a NPT integrator.

    """
    def __init__(self, add_op):
        if not isinstance(add_op, AddNPTIntegrator):
            raise TypeError('Addition operation is not AddNPTIntegrator.')
        self.add_op = add_op

    def to_commands(self, sim):
        cmds = ['unfix {idx}'.format(idx=self.add_op._fix)]

        return cmds

class AddNVTIntegrator(LAMMPSOperation):
    r"""NVT integration via Nos\'e-Hoover thermostat.

    The simulation ensemble must be NVT.

    Parameters
    ----------
    dt : float
        Time step size for each simulation iteration.
    tau_T : float
        Coupling constant for the thermostat.

    """
    def __init__(self, dt, tau_T):
        self.tau_T = tau_T

        self._fix = super().new_fix_id()

    def to_commands(self, sim):
        if not sim.ensemble.aka('NVT'):
            raise ValueError('Simulation ensemble is not NVT.')
        cmds = ['fix {idx} {group_idx} nvt temp {Tstart} {Tstop} {Tdamp}'.format(idx=self._fix,
                                                                                 group_idx='all',
                                                                                 Tstart=sim.ensemble.T,
                                                                                 Tstop=sim.ensemble.T,
                                                                                 Tdamp=self.tau_T)]

        return cmds

class RemoveNVTIntegrator(LAMMPSOperation):
    """Removes the NVT integrator operation.

    Parameters
    ----------
    add_op : :py:class:`AddNVTIntegrator`
        The integrator addition operation to be removed.

    Raises
    ------
    TypeError
        If the specified addition operation is not a NVT integrator.

    """
    def __init__(self, add_op):
        if not isinstance(add_op, AddNVTIntegrator):
            raise TypeError('Addition operation is not AddNVTIntegrator.')
        self.add_op = add_op

    def to_commands(self, sim):
        cmds = ['unfix {idx}'.format(idx=self.add_op._fix)]

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

        return cmds

class AddEnsembleAnalyzer(LAMMPSOperation):
    """Analyzes the simulation ensemble and rdf at specified timestep intervals.

    The simulation ensemble must have constant N, and only one LAMMPS
    :py:class:AddEnsembleAnalyzer` can be initialized at a time.

    Parameters
    ----------
    check_thermo_every : int
        Interval of time steps at which to log thermodynamic properties of the simulation.
    check_rdf_every : int
        Interval of time steps at which to log the rdf of the simulation.
    rdf_dr : float
        The width (in units *r*) of a bin in the histogram of the rdf.

    """
    def __init__(self, check_thermo_every, check_rdf_every, rdf_dr):
        self.check_thermo_every = check_thermo_every
        self.check_rdf_every = check_rdf_every
        self.rdf_dr = rdf_dr

    def to_commands(self, sim):
        if not all([sim.ensemble.constant['N'][t] for t in sim.ensemble.types]):
            return ValueError('This analyzer requires constant N.')

        # check that IDs reserved for analysis do not yet exist
        reserved_ids = (('fix','thermo_avg'),
                        ('compute','rdf'),
                        ('fix','rdf_avg'),
                        ('variable','T'),
                        ('variable','P'),
                        ('variable','Lx'),
                        ('variable','Ly'),
                        ('variable','Lz'),
                        ('variable','xy'),
                        ('variable','xz'),
                        ('variable','yz'))
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
                'variable ensemble_yz equal yz',
                ('fix ensemble_thermo_avg all ave/time {every} 1 {every}'
                 ' v_ensemble_T v_ensemble_P'
                 ' v_ensemble_Lx v_ensemble_Ly v_ensemble_Lz'
                 ' v_ensemble_xy v_ensemble_xz v_ensemble_yz'
                 ' mode scalar ave running'
                 ' file {filename} overwrite format "%.16e"').format(every=self.check_thermo_every,
                                                                     filename=sim[self].thermo_file)
                ]

        # pair distribution function
        rmax = sim.potentials.pair.r[-1]
        sim[self].num_bins = np.round(rmax/self.rdf_dr).astype(int)
        sim[self].rdf_file = sim.directory.file('lammps_rdf.dat')
        sim[self].rdf_pairs = tuple(sim.ensemble.pairs)
        # string format lammps arguments based on pairs
        # _pairs is the list of all pairs by LAMMPS type id, in ensemble order
        # _computes is the RDF values for each pair, with the r bin centers prepended
        _pairs = []
        _computes = ['c_ensemble_rdf[1]']
        for idx,(i,j) in enumerate(sim[self].rdf_pairs):
            _pairs.append('{} {}'.format(sim.type_map[i],sim.type_map[j]))
            _computes.append('c_ensemble_rdf[{}]'.format(2*(idx+1)))
        cmds += ['compute ensemble_rdf all rdf {bins} {pairs}'.format(bins=sim[self].num_bins,pairs=' '.join(_pairs)),
                 ('fix ensemble_rdf_avg all ave/time {every} 1 {every}'
                  ' {computes} mode vector ave running off 1'
                  ' file {filename} overwrite format "%.16e"').format(every=self.check_rdf_every,
                                                                      computes=' '.join(_computes),
                                                                      filename=sim[self].rdf_file)
                ]

        return cmds

    def extract_ensemble(self, sim):
        """Creates an ensemble with the averaged thermodynamic properties and rdf.

        Parameters
        ----------
        sim : :py:class:`Simulation`
            The simulation object.

        Returns
        -------
        :py:class:`Ensemble`
            Ensemble with averaged thermodynamic properties and rdf.

        """
        ens = sim.ensemble.copy()
        ens.clear()

        # extract thermo properties
        # we skip the first 2 rows, which are LAMMPS junk, and slice out the timestep from col. 0
        thermo = np.loadtxt(sim[self].thermo_file,skiprows=2)[1:]
        ens.T = thermo[0]
        ens.P = thermo[1]
        ens.V = TriclinicBox(Lx=thermo[2],Ly=thermo[3],Lz=thermo[4],
                             xy=thermo[5],xz=thermo[6],yz=thermo[7],
                             convention=TriclinicBox.Convention.LAMMPS)

        # extract rdfs
        # LAMMPS injects a column for the row index, so we start at column 1 for r
        # we skip the first 4 rows, which are LAMMPS junk, and slice out the first column
        rdf = np.loadtxt(sim[self].rdf_file,skiprows=4)[:,1:]
        for i,pair in enumerate(sim[self].rdf_pairs):
            ens.rdf[pair] = RDF(rdf[:,0],rdf[:,i+1])

        return ens
