r"""
Objective functions
===================

An objective function is the quantity to be minimized in an optimization problem,
by adjusting the variables on which the function depends.

This function, :math:`f`, is a scalar value that is defined as a function of :math:`n`
problem :class:`~relentless.variable.IndependentVariable`\s
:math:`\mathbf{x}=\left[x_1,\ldots,x_n\right]`.

The value of the function, :math:`f\left(\mathbf{x}\right)` is specified.
The gradient is also specified for all of the design variables:

    .. math::

        \nabla f\left(\mathbf{x}\right) = \left[\frac{\partial f}{\partial x_1},
                                                \ldots,
                                                \frac{\partial f}{\partial x_n}\right]

The following objective functions have been implemented:

.. autosummary::
    :nosignatures:

    RelativeEntropy

.. rubric:: Developer notes

To implement your own objective function, create a class that derives from
:class:`ObjectiveFunction` and define the required properties and methods.

.. autosummary::
    :nosignatures:

    ObjectiveFunction
    ObjectiveFunctionResult

.. autoclass:: ObjectiveFunction
    :member-order: bysource
    :members: compute

.. autoclass:: ObjectiveFunctionResult
    :member-order: bysource
    :members: variables,
        value,
        gradient,
        directory

.. autoclass:: RelativeEntropy
    :member-order: bysource
    :members: compute,
        compute_gradient,
        target

"""

import abc
import json
import pathlib
import tempfile

import freud
import gsd.hoomd
import numpy

from relentless import data, math, mpi
from relentless.model import Ensemble, extent, variable
from relentless.simulate.analyze import EnsembleAverage, WriteTrajectory


class ObjectiveFunction(abc.ABC):
    r"""Abstract base class for the optimization objective function.

    An :class:`ObjectiveFunction` defines the objective function parametrized on
    one or more adjustable :class:`~relentless.variable.IndependentVariable`\s.
    The function must also have a defined value and gradient for all values of its
    parameters.

    """

    @abc.abstractmethod
    def compute(self, variables, directory=None):
        """Evaluate the value and gradient of the objective function.

        Parameters
        ----------
        variables : :class:`~relentless.variable.Variable` or tuple
            Variables to record in result.
        directory : str or :class:`~relentless.data.Directory`
            The ouptut directory. In addition to simulation output, the pair
            potential design variables at the time of computation are saved
            (defaults to ``None``).

        Returns
        -------
        :class:`ObjectiveFunctionResult`
            The result of the function.

        """
        pass


class ObjectiveFunctionResult:
    """Class storing the value and gradient of a :class:`ObjectiveFunction`.

    Parameters
    ----------
    variables : :class:`~relentless.variable.Variable` or tuple
        Variables to stash values (defaults to ``None``).
    value : float
        The value of the objective function (defaults to ``None``).
    gradient : dict
        The gradient of the objective function. Each partial derivative is
        keyed on the :class:`~relentless.variable.IndependentVariable`
        with respect to which it is taken (defaults to ``None``).
    directory : :class:`~relentless.data.Directory`
        Directory holding written output associated with result. Setting
        a value of ``None`` indicates no written output (defaults to ``None``).

    Raises
    ------
    KeyError
        If both ``variables`` and ``gradient`` are defined but their keys
        don't match.

    """

    def __init__(self, variables=None, value=None, gradient=None, directory=None):
        self.variables = variables
        self.value = value
        self.gradient = gradient
        self.directory = directory

    @property
    def variables(self):
        """:class:`~relentless.math.KeyedArray`: Recorded variables of the
        :class:`ObjectiveFunction`."""
        return getattr(self, "_variables", None)

    @variables.setter
    def variables(self, value):
        value = variable.graph.check_variables_and_types(value, variable.Variable)
        if len(value) > 0:
            variables_ = math.KeyedArray(keys=value)
            variables_.update({x: x.value for x in value})
        else:
            variables_ = None
        self._assert_keys_match(variables_, self.gradient)
        self._variables = variables_

    @property
    def value(self):
        """float: The value of the evaluated objective function."""
        return getattr(self, "_value", None)

    @value.setter
    def value(self, x):
        self._value = x

    @property
    def gradient(self):
        """:class:`~relentless.math.KeyedArray`: The gradient of the objective
        function, keyed on its design variables."""
        return getattr(self, "_gradient", None)

    @gradient.setter
    def gradient(self, value):
        if value is not None:
            gradient_ = math.KeyedArray(keys=value.keys())
            gradient_.update(value)
        else:
            gradient_ = None
        self._assert_keys_match(self.variables, gradient_)
        self._gradient = gradient_

    @property
    def directory(self):
        """:class:`~relentless.data.Directory` Directory holding written output."""
        return getattr(self, "_directory", None)

    @directory.setter
    def directory(self, value):
        if value is not None:
            value = data.Directory.cast(value, create=mpi.world.rank_is_root)
            mpi.world.barrier()
        self._directory = value

    def save(self, filename):
        r"""Save the result as a JSON file.

        Parameters
        ----------
        filename : str
            The name of the file to save data in.

        """
        data = {
            "variables": {x.name: v for x, v in self.variables.items()},
            "value": self.value,
            "gradient": {x.name: v for x, v in self.gradient.items()},
            "directory": self.directory.path if self.directory is not None else None,
        }

        # dump data to json file
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

    def _assert_keys_match(self, vars, grad):
        """Assert that the keys of the variables and gradient match.

        Parameters
        ----------
        vars : dict
            Variable dictionary-like object
        grad : dict
            Gradient dictionary-like object.

        Raises
        ------
        AssertionError
            If the keys of ``vars`` and ``grad`` do not match.

        """
        if vars is not None and grad is not None:
            if vars.keys() != grad.keys():
                raise AssertionError("Variable and gradient keys do not match!")


class RelativeEntropy(ObjectiveFunction):
    r"""Relative entropy.

    The relative entropy :math:`S_{\rm rel}` (or Kullback-Leibler divergence)
    quantifies the overlap of two probability distributions. For a known target
    statistical mechanical ensemble having distribution :math:`p_0` and a simulated
    model ensemble having distribution :math:`p` and parametrized on a set of design
    variables :math:`\mathbf{x}`, the relative entropy from the model to the target is:

    .. math::

        S_{\rm rel} =
            -\int d\Gamma p_0(\Gamma)\ln\left(\frac{p(\Gamma)}{p_0(\Gamma)}\right)

    where :math:`\Gamma` is an element of phase space. The relative entropy is
    zero when the two ensembles overlap completely, and it is positive otherwise.
    Hence, minimization of :math:`S_{\rm rel}` can be used to find parameters of
    a model that reproduce a known target ensemble.

    The value of the relative entropy is not readily determined in  molecular
    simulations, so this :class:`ObjectiveFunction` does not return a value.
    However, the gradient of the relative entropy with respect to the design
    variables :math:`\mathbf{x}` is much easier to compute as ensemble averages.
    Currently, the :class:`RelativeEntropy` objective function supports only
    :class:`~relentless.potential.pair.PairPotential` interactions. These interactions
    are characterized by :math:`g_{ij}(r)`, an :class:`~relentless.ensemble.RDF`
    for each pair of interacting types :math:`(i,j)` in each
    :class:`~relentless.ensemble.Ensemble`. The gradient of :math:`S_{\rm rel}` is then:

    .. math::

        \nabla_\mathbf{x} S_{\rm rel} = -\frac{1}{2} \sum_{i,j} \int dr 4\pi r^2
            \left[\frac{\beta N_i N_j}{V} g_{ij}(r) \right.
            \left. -\frac{\beta_0 N_{i,0} N_{j,0}}{V_0} g_{ij,0}(r)\right]
            \nabla_\mathbf{x} u_{ij}(r)}

    where :math:`\beta=1/(k_{\rm B}T)`, :math:`N_i` is the number of particles
    of type :math:`i`, :math:`V` is the extent, and :math:`u_{ij}(r)` is the pair
    potential in the *model* ensemble. The corresponding properties of the *target*
    ensemble are denoted with subscript :math:`0`.

    :math:`S_{\rm rel}` is extensive as written, meaning that it depends on the
    size of the system. This can be undesirable for optimization because it means
    certain hyperparameters are system-dependent, so the default behavior
    is to normalize :math:`s_{\rm rel}=S_{\rm rel}/V_0`. To use the extensive
    relative entropy set ``extensive=True``.

    Parameters
    ----------
    target : :class:`~relentless.ensemble.Ensemble`
        The target ensemble (must have specified ``V`` and ``N``).
    simulation : :class:`~relentless.simulate.simulate.Simulation`
        The simulation engine to use, with specified simulation operations.
    potentials : :class:`~relentless.simulate.simulate.Potentials`
        The pair potentials to use in the simulations.
    thermo : :class:`~relentless.simulate.simulate.SimulationOperation`
        The thermodynamic analyzer operation for the simulation ensemble and rdf
        (usually :class:`~relentless.simulate.simulate.AddEnsembleAnalyzer`).
        The model ensemble will be extracted from this operation.
    extensive : bool
        Specification of whether the relative entropy is extensive (defaults to
        ``False``).

    """

    def __init__(self, target, simulation, potentials, thermo, extensive=False, T=None):
        self.target = target
        self.simulation = simulation
        self.potentials = potentials
        self.thermo = thermo
        self.extensive = extensive
        self.T = T

    def compute(self, variables, directory=None):
        r"""Evaluate the value and gradient of the relative entropy function.

        The value of the relative entropy is not computed, but the gradient is.
        Calculating the gradient requires running a simulation, which may be
        computationally expensive.

        Optionally, a directory can be specified to write the simulation output
        as defined in :meth:`~relentless.simulate.simulate.Simulation.run()`,
        namely the values of the pair potential design variables,
        which are written to ``pair_potential.i.json`` for the :math:`i`\th pair
        potential, and the :class:`~relentless.optimize.ObjectiveFunctionResult`,
        which is written to ``result.json``.

        Parameters
        ----------
        variables : :class:`~relentless.variable.Variable` or tuple
            Variables with respect to which to compute gradient.
        directory : str or :class:`~relentless.data.Directory`
            The output directory

        Returns
        -------
        :class:`ObjectiveFunctionResult`
            The result, which has unknown value ``None`` and known gradient.

        """
        # a directory is needed for the simulation, so create one if we don't have one
        if directory is None:
            # create directory and synchronize
            if mpi.world.rank_is_root:
                tmp = tempfile.TemporaryDirectory()
                directory = tmp.name
            else:
                tmp = None
            directory = mpi.world.bcast(directory)
            directory_is_tmp = True
        else:
            tmp = None
            directory_is_tmp = False
        directory = data.Directory.cast(directory, create=mpi.world.rank_is_root)
        mpi.world.barrier()

        # write the pair potential parameters *before* the run
        if not directory_is_tmp:
            if mpi.world.rank_is_root:
                for n, p in enumerate(self.potentials.pair.potentials):
                    p.save(directory.file("pair_potential.{}.json".format(n)))
            mpi.world.barrier()

        # run simulation and use result to compute gradient
        try:
            sim = self.simulation.run(self.potentials, directory)
            if self._use_trajectory(self.target, self.thermo):
                sim_ens = sim.directory.file(self.thermo.filename)
            else:
                if self.thermo.rdf is None:
                    raise ValueError(
                        "EnsembleAverage needs to compute RDF, specify parameters."
                    )
                sim_ens = sim[self.thermo]["ensemble"]
        finally:
            mpi.world.barrier()
            if tmp is not None:
                tmp.cleanup()

        # compute gradient and result
        # relative entropy *value* is None
        gradient = None
        if mpi.world.rank_is_root:
            if self._use_trajectory(self.target, self.thermo):
                gradient = self._compute_gradient_direct_average(sim_ens, variables)
            else:
                gradient = self.compute_gradient(sim_ens, variables)
        mpi.world.bcast(gradient)
        result = ObjectiveFunctionResult(
            variables, None, gradient, directory if not directory_is_tmp else None
        )

        # optionally write ensemble and result *after* the simulation
        if not directory_is_tmp:
            if mpi.world.rank_is_root:
                result.save(directory.file("result.json"))
            mpi.world.barrier()

        return result

    def _compute_gradient_direct_average(self, sim_traj, variables):
        r"""Computes the relative entropy gradient for an ensemble.

        Parameters
        ----------
        sim_traj : :class:`string`
            Path to gsd trajectory file.
        variables : :class:`~relentless.variable.Variable` or tuple
            Variables with respect to which to compute gradient.

        Returns
        -------
        :class:`~relentless.math.KeyedArray`
            The gradient, keyed on the ``variables``.

        """

        # load the target ensemble trajectory file
        traj_tgt = self.target
        dvars = variable.graph.check_variables_and_types(variables, variable.Variable)

        with gsd.hoomd.open(traj_tgt, "r") as traj:
            pair_types_tgt = traj[0].particles.types

            bond_types_tgt = None
            if traj[0].bonds.N != 0:
                bond_types_tgt = traj[0].bonds.types

            angle_types_tgt = None
            if traj[0].angles.N != 0:
                angle_types_tgt = traj[0].angles.types

            dihedral_types_tgt = None
            if traj[0].dihedrals.N != 0:
                dihedral_types_tgt = traj[0].dihedrals.types

            V_tgt = 0
            for snap in traj:
                Lx, Ly, Lz, xy, xz, yz = snap.configuration.box
                V_tgt += extent.TriclinicBox(Lx, Ly, Lz, xy, xz, yz).extent / len(traj)

        with gsd.hoomd.open(sim_traj, "r") as traj:
            pair_types_sim = traj[0].particles.types

            bond_types_sim = None
            if traj[0].bonds.N != 0:
                bond_types_sim = traj[0].bonds.types

            angle_types_sim = None
            if traj[0].angles.N != 0:
                angle_types_sim = traj[0].angles.types

            dihedral_types_sim = None
            if traj[0].dihedrals.N != 0:
                dihedral_types_sim = traj[0].dihedrals.types

            # calculate T if not provided
            if self.T is None:
                if snap.particles.mass is None or snap.particles.velocity is None:
                    raise ValueError(
                        "Temperature not provided and cannot be calculated "
                        "from trajectory."
                    )
                T_avg = 0
                for snap in traj:
                    T_avg += numpy.sum(
                        snap.particles.mass
                        * numpy.linalg.norm(snap.particles.velocity, axis=1) ** 2
                        / (3 * snap.particles.N * self.potentials.kB)
                    ) / len(traj)
                self.T = T_avg

        if pair_types_tgt != pair_types_sim:
            raise ValueError(
                "Particle types in target and simulation trajectories do not match."
            )
        if bond_types_tgt != bond_types_sim:
            raise ValueError(
                "Bond types in target and simulation trajectories do not match."
            )
        if angle_types_tgt != angle_types_sim:
            raise ValueError(
                "Angle types in target and simulation trajectories do not match."
            )
        if dihedral_types_tgt != dihedral_types_sim:
            raise ValueError(
                "Dihedral types in target and simulation trajectories do not match."
            )

        # normalization to extensive or intensive as specified
        norm_factor = V_tgt if not self.extensive else 1.0
        gradient = (
            self._calc_ensemble_average_dvar_dlambda(traj_tgt, dvars)
            - self._calc_ensemble_average_dvar_dlambda(sim_traj, dvars)
        ) / (norm_factor * self.potentials.kB * self.T)
        return gradient

    def _calc_ensemble_average_dvar_dlambda(self, trajectory, variables):
        gradient = math.KeyedArray(keys=variables)
        for var in variables:
            gradient[var] = 0

        with gsd.hoomd.open(trajectory, "r") as traj:
            # loop through the trajectory and calculate ensemble average
            box = freud.box.Box.from_box(traj[0].configuration.box)
            for snap in traj:
                pos = snap.particles.position
                aq = freud.locality.AABBQuery(box, pos)

                neighbors = aq.query(
                    pos,
                    dict(
                        mode="ball",
                        r_max=self.potentials.pair.stop,
                        exclude_ii=True,
                    ),
                ).toNeighborList()
                type_masks = {}
                for i in snap.particles.types:
                    type_masks[i] = snap.particles.typeid == snap.particles.types.index(
                        i
                    )

                bonded_exclusions = self.potentials.pair.exclusions
                if bonded_exclusions is not None:
                    if (
                        "1-2" in bonded_exclusions
                        and snap.bonds.N != 0
                        and len(neighbors[:]) > 0
                    ):
                        bonds = numpy.vstack(
                            [snap.bonds.group, numpy.flip(snap.bonds.group, axis=1)],
                        )

                        bond_exclusion_filter = EnsembleAverage._cantor_pairing(
                            self, bonds, neighbors
                        )

                        neighbors.filter(bond_exclusion_filter)
                    if (
                        "1-3" in bonded_exclusions
                        and snap.angles.N != 0
                        and len(neighbors[:]) > 0
                    ):
                        angles = numpy.vstack(
                            [snap.angles.group, numpy.flip(snap.angles.group, axis=1)],
                        )

                        angle_exclusion_filter = EnsembleAverage._cantor_pairing(
                            self, angles[:, (0, -1)], neighbors
                        )

                        neighbors.filter(angle_exclusion_filter)
                    if (
                        "1-4" in bonded_exclusions
                        and snap.dihedrals.N != 0
                        and len(neighbors[:]) > 0
                    ):
                        dihedrals = numpy.vstack(
                            [
                                snap.dihedrals.group,
                                numpy.flip(snap.dihedrals.group, axis=1),
                            ],
                        )

                        dihedral_exclusion_filter = EnsembleAverage._cantor_pairing(
                            self, dihedrals[:, (0, -1)], neighbors
                        )

                        neighbors.filter(dihedral_exclusion_filter)

                filter_j_gt_i = neighbors[:, 1] > neighbors[:, 0]
                neighbors.filter(filter_j_gt_i)
                # pair contributions to the gradient
                for i in snap.particles.types:
                    for j in snap.particles.types:
                        filter_ij = numpy.logical_and(
                            type_masks[i][neighbors[:, 0]],
                            type_masks[j][neighbors[:, 1]],
                        )
                        for var in variables:
                            gradient[var] += numpy.sum(
                                self.potentials.pair.derivative(
                                    (i, j), var, x=neighbors.distances[filter_ij]
                                )
                            ) / len(traj)

                # bond contributions to the gradient
                if snap.bonds.N != 0:
                    bond_type_map = {type: i for i, type in enumerate(snap.bonds.types)}
                    for i in snap.bonds.types:
                        bond_type_filter = bond_type_map[i] == snap.bonds.typeid

                        bonds = snap.bonds.group[bond_type_filter]

                        # Get positions for all pairs of bonded particles
                        pos_1 = pos[bonds[:, 0]]
                        pos_2 = pos[bonds[:, 1]]

                        # Calculate distances
                        dr = numpy.linalg.norm(pos_2 - pos_1, axis=1)

                        for var in variables:
                            rs = self.potentials.bond.linear_space
                            dus = self.potentials.bond.derivative(i, var)
                            dudvar = math.AkimaSpline(rs, dus)
                            gradient[var] += numpy.sum(dudvar(dr)) / len(traj)

                # angle contributions to the gradient
                if snap.angles.N != 0:
                    angle_type_map = {
                        type: i for i, type in enumerate(snap.angles.types)
                    }
                    for i in snap.angles.types:
                        angle_type_filter = angle_type_map[i] == snap.angles.typeid

                        angles = snap.angles.group[angle_type_filter]

                        # get positions for all triplets of bonded particles
                        pos_1 = pos[angles[:, 0]]
                        pos_2 = pos[angles[:, 1]]
                        pos_3 = pos[angles[:, 2]]

                        # calculate angles
                        r_12 = pos_2 - pos_1
                        r_23 = pos_3 - pos_2

                        # use einsum for row-wise dot product
                        dtheta = numpy.arccos(
                            -numpy.einsum("ij,ij->i", r_12, r_23)
                            / (numpy.linalg.norm(r_12) * numpy.linalg.norm(r_23))
                        )

                        for var in variables:
                            rs = self.potentials.angle.linear_space
                            dus = self.potentials.angle.derivative(i, var)
                            dudvar = math.AkimaSpline(rs, dus)
                            gradient[var] += numpy.sum(dudvar(dtheta)) / len(traj)

                # dihedral contributions to the gradient
                if snap.dihedrals.N != 0:
                    dihedral_type_map = {
                        type: i for i, type in enumerate(snap.dihedrals.types)
                    }
                    for i in snap.dihedrals.types:
                        dihedral_type_filter = (
                            dihedral_type_map[i] == snap.dihedrals.typeid
                        )

                        dihedrals = snap.dihedrals.group[dihedral_type_filter]

                        # get positions for all quadruplets of bonded particles
                        pos_1 = pos[dihedrals[:, 0]]
                        pos_2 = pos[dihedrals[:, 1]]
                        pos_3 = pos[dihedrals[:, 2]]
                        pos_4 = pos[dihedrals[:, 3]]

                        # calculate dihedrals
                        r_12 = pos_2 - pos_1
                        r_23 = pos_3 - pos_2
                        r_34 = pos_4 - pos_3

                        cross_12_23 = numpy.cross(r_12, r_23)
                        cross_23_34 = numpy.cross(r_23, r_34)
                        dphi = numpy.arccos(
                            numpy.einsum("ij,ij->i", cross_12_23, cross_23_34)
                            / (
                                numpy.linalg.norm(cross_12_23)
                                * numpy.linalg.norm(cross_23_34)
                            )
                        )
                        for var in variables:
                            rs = self.potentials.dihedral.linear_space
                            dus = self.potentials.dihedral.derivative(i, var)
                            dudvar = math.AkimaSpline(rs, dus)
                            gradient[var] += numpy.sum(dudvar(dphi)) / len(traj)
        return gradient

    def compute_gradient(self, ensemble, variables):
        r"""Computes the relative entropy gradient for an ensemble.

        Parameters
        ----------
        ensemble : :class:`~relentless.ensemble.Ensemble`
            The ensemble for which to evaluate the gradient.
        variables : :class:`~relentless.variable.Variable` or tuple
            Variables with respect to which to compute gradient.

        Returns
        -------
        :class:`~relentless.math.KeyedArray`
            The gradient, keyed on the ``variables``.

        """
        # compute the relative entropy gradient by integration
        g_tgt = self.target.rdf
        g_sim = ensemble.rdf
        dvars = variable.graph.check_variables_and_types(variables, variable.Variable)
        gradient = math.KeyedArray(keys=dvars)

        # make sure RDF has been computed
        for pair in g_tgt:
            if g_sim.get(pair, None) is None:
                raise KeyError(f"RDF not computed for pair {pair}")

        for var in dvars:
            update = 0
            for i, j in g_tgt:
                rs = self.potentials.pair.linear_space
                us = self.potentials.pair.energy((i, j))
                dus = self.potentials.pair.derivative((i, j), var)

                # only count (continuous range of) finite values
                flags = numpy.isinf(us)
                first_finite = 0
                while flags[first_finite] and first_finite < len(rs):
                    first_finite += 1
                rs = rs[first_finite:]
                dus = dus[first_finite:]
                if first_finite == len(rs):
                    continue

                # interpolate derivative wrt design variable with r
                dudvar = math.AkimaSpline(rs, dus)

                # find common domain to compare rdfs
                r0 = max(
                    g_sim[i, j].table[0, 0], g_tgt[i, j].table[0, 0], dudvar.table[0, 0]
                )
                r1 = min(
                    g_sim[i, j].table[-1, 0],
                    g_tgt[i, j].table[-1, 0],
                    dudvar.table[-1, 0],
                )
                sim_dr = numpy.min(numpy.diff(g_sim[i, j].table[:, 0]))
                tgt_dr = numpy.min(numpy.diff(g_tgt[i, j].table[:, 0]))
                dudvar_dr = numpy.min(numpy.diff(rs))
                dr = min(sim_dr, tgt_dr, dudvar_dr)
                r = numpy.arange(r0, r1 + 0.5 * dr, dr)

                # normalization to extensive or intensive as specified
                norm_factor = self.target.V.extent if not self.extensive else 1.0

                # take integral by trapezoidal rule
                sim_factor = (
                    ensemble.N[i]
                    * ensemble.N[j]
                    / (
                        self.potentials.kB
                        * ensemble.T
                        * ensemble.V.extent
                        * norm_factor
                    )
                )
                tgt_factor = (
                    self.target.N[i]
                    * self.target.N[j]
                    / (
                        self.potentials.kB
                        * self.target.T
                        * self.target.V.extent
                        * norm_factor
                    )
                )
                mult = (
                    1 if i == j else 2
                )  # 1 if same, otherwise need i,j and j,i contributions
                if isinstance(self.target.V, extent.Volume):
                    geo_prefactor = 4 * numpy.pi * r**2
                elif isinstance(self.target.V, extent.Area):
                    geo_prefactor = 2 * numpy.pi * r
                else:
                    raise ValueError(
                        "Geometric integration factor unknown for extent type"
                    )
                y = (
                    -0.5
                    * mult
                    * geo_prefactor
                    * (sim_factor * g_sim[i, j](r) - tgt_factor * g_tgt[i, j](r))
                    * dudvar(r)
                )
                update += math._trapezoid(y, r)

            gradient[var] = update

        return gradient

    @property
    def target(self):
        r""":class:`~relentless.ensemble.Ensemble`: The target ensemble. Must have
        both ``V`` and ``N`` parameters set."""
        return self._target

    @target.setter
    def target(self, value):
        if isinstance(value, Ensemble):
            if value.V is None or value.N is None:
                raise ValueError("The target ensemble must have both V and N set.")
        elif isinstance(value, str):
            file_suffix = pathlib.Path(value).suffix

            if not file_suffix == ".gsd":
                raise ValueError("Target must be a gsd trajectory file.")
        self._target = value

    def _use_trajectory(self, target, thermo):
        r"""Check if the target and thermo are an ensemble and ensemble average
        or string and WriteTrajectory.

        Parameters
        ----------
        target : :class:`~relentless.ensemble.Ensemble` or `str`.
            The target ensemble or trajectory file.

        thermo : :class:`~relentless.simulate.analyze.EnsembleAverage` or
            :class:`~relentless.simulate.analyze.WriteTrajectory`
            The thermodynamic analyzer operation.

        Returns
        -------
        bool
            ``True`` if the target and thermo are string and WriteTrajectory,
            ``False`` if the target and thermo are an ensemble and ensemble average,

        Raises
        ------
        TypeError
            If the target and thermo are not an ensemble and ensemble average or
            string and WriteTrajectory.


        """
        if isinstance(target, str) and isinstance(thermo, WriteTrajectory):
            return True
        elif isinstance(target, Ensemble) and isinstance(thermo, EnsembleAverage):
            return False
        else:
            raise TypeError(
                "RelativeEntropy target and thermo must be either an Ensemble"
                " and EnsembleAverage or a string and WriteTrajectory"
            )
