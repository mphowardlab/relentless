from . import simulate

class MinimizeEnergy(simulate.GenericOperation):
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
        super().__init__(energy_tolerance, force_tolerance, max_iterations, options)

class RunBrownianDynamics(simulate.GenericOperation):
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
        super().__init__(steps, timestep, T, friction, seed, analyzers)

class RunLangevinDynamics(simulate.GenericOperation):
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
        super().__init__(steps, timestep, T, friction, seed, analyzers)

class RunMolecularDynamics(simulate.GenericOperation):
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
        super().__init__(steps, timestep, thermostat, barostat, analyzers)

class Thermostat:
    """Thermostat.

    Controls the temperature of the simulation by modifying the particle
    velocities according to a given scheme.

    Parameters
    ----------
    T : float
        Target temperature.

    """
    def __init__(self, T):
        self.T = T

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

