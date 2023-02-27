from . import simulate


class MinimizeEnergy(simulate.DelegatedSimulationOperation):
    """Perform energy minimization on a configuration.

    Description.

    Parameters
    ----------
    energy_tolerance : float
        Energy convergence criterion.
    force_tolerance : float
        Force convergence criterion.
    max_iterations : int
        Maximum number of iterations to run the minimization.
    options : dict
        Additional options for energy minimizer (defaults to ``None``).

    """

    def __init__(self, energy_tolerance, force_tolerance, max_iterations, options=None):
        super().__init__(None)
        self.energy_tolerance = energy_tolerance
        self.force_tolerance = force_tolerance
        self.max_iterations = max_iterations
        self.options = options

    def _make_delegate(self, sim):
        return self._get_delegate(
            sim,
            energy_tolerance=self.energy_tolerance,
            force_tolerance=self.force_tolerance,
            max_iterations=self.max_iterations,
            options=self.options,
        )


class _Integrator(simulate.DelegatedSimulationOperation):
    """Base molecular dynamics integrator.

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


class RunBrownianDynamics(_Integrator):
    """Perform a Brownian dynamics simulation.

    Description.

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

    def __init__(self, steps, timestep, T, friction, seed, analyzers=None):
        super().__init__(steps, timestep, analyzers)
        self.T = T
        self.friction = friction
        self.seed = seed

    def _make_delegate(self, sim):
        return self._get_delegate(
            sim,
            steps=self.steps,
            timestep=self.timestep,
            T=self.T,
            friction=self.friction,
            seed=self.seed,
            analyzers=self.analyzers,
        )


class RunLangevinDynamics(_Integrator):
    """Perform a Langevin dynamics simulation.

    Description.

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

    def __init__(self, steps, timestep, T, friction, seed, analyzers=None):
        super().__init__(steps, timestep, analyzers)
        self.T = T
        self.friction = friction
        self.seed = seed

    def _make_delegate(self, sim):
        return self._get_delegate(
            sim,
            steps=self.steps,
            timestep=self.timestep,
            T=self.T,
            friction=self.friction,
            seed=self.seed,
            analyzers=self.analyzers,
        )


class RunMolecularDynamics(_Integrator):
    """Perform a molecular dynamics simulation.

    The Verlet-style integrator is used to implement classical molecular
    dynamics equations of motion. The integrator may optionally accept a
    :class:`Thermostat` and a :class:`Barostat` for temperature and pressure
    control, respectively. Depending on the :class:`Simulation` engine, not
    all combinations of ``thermostat`` and ``barostat`` may be allowed. Refer
    to the specific documentation for the engine you plan to use if you are
    unsure or obtain an error for your chosen combination.

    Parameters
    ----------
    steps : int
        Number of simulation time steps.
    timestep : float
        Simulation time step.
    thermostat : :class:`Thermostat`
        Thermostat used for integration (defaults to ``None``).
    barostat : :class:`Barostat`
        Barostat used for integration (defaults to ``None``).
    analyzers : :class:`~relentless.simulate.AnalysisOperation` or list
        Analysis operations to perform with run (defaults to ``None``).

    """

    def __init__(self, steps, timestep, thermostat=None, barostat=None, analyzers=None):
        super().__init__(steps, timestep, analyzers)
        self.thermostat = thermostat
        self.barostat = barostat

    def _make_delegate(self, sim):
        return self._get_delegate(
            sim,
            steps=self.steps,
            timestep=self.timestep,
            thermostat=self.thermostat,
            barostat=self.barostat,
            analyzers=self.analyzers,
        )


class Thermostat:
    """Thermostat.

    Controls the temperature of the simulation by modifying the particle
    velocities according to a given scheme.

    Either 1 or 2 values can be specified for ``T``. If 1 value is given, the
    temperature will be held constant. If 2 values are given, the temperature
    will be varied linearly (annealed) between the values.

    Parameters
    ----------
    T : float or tuple
        Target temperature(s).

    """

    def __init__(self, T):
        self.T = T

    @property
    def T(self):
        """float or tuple: Temperature set point(s)."""
        return self._T

    @T.setter
    def T(self, value):
        try:
            num_temps = len(value)
            if num_temps > 2:
                raise ValueError("Only up to 2 temperatures supported")

            if num_temps == 2:
                self._T = tuple(value)
                self._anneal = True
            elif num_temps == 1:
                self._T = value[0]
                self._anneal = False
            else:
                raise ValueError("At least 1 temperature must be specified")
        except TypeError:
            self._T = value
            self._anneal = False

    @property
    def anneal(self):
        """bool: True if temperature annealing will be done."""
        return self._anneal


class BerendsenThermostat(Thermostat):
    """Berendsen thermostat.

    Description.

    Parameters
    ----------
    T : float
        Target temperature.
    tau : float
        Coupling time constant.

    """

    def __init__(self, T, tau):
        super().__init__(T)
        self.tau = tau


class NoseHooverThermostat(Thermostat):
    """Nosé-Hoover thermostat.

    Parameters
    ----------
    T : float
        Target temperature.
    tau : float
        Coupling time constant.

    """

    def __init__(self, T, tau):
        super().__init__(T)
        self.tau = tau


class Barostat:
    """Barostat.

    Controls the pressure of the simulation by scaling the box volume
    according to a given scheme.

    Parameters
    ----------
    P : float
        Target pressure.

    """

    def __init__(self, P):
        self.P = P


class BerendsenBarostat(Barostat):
    """Berendsen barostat.

    Description.

    Parameters
    ----------
    P : float
        Target pressure.
    tau : float
        Coupling time constant.

    """

    def __init__(self, P, tau):
        super().__init__(P)
        self.tau = tau


class MTKBarostat(Barostat):
    """MTK barostat.

    Description

    Parameters
    ----------
    P : float
        Target pressure.
    tau : float
        Coupling time constant.

    """

    def __init__(self, P, tau):
        super().__init__(P)
        self.tau = tau
