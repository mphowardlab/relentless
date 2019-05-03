from . import utils

class Engine(object):
    def __init__(self,env):
        self.env = env

    def setup(self, step):
        pass

    def run(self, step, potential):
        """ Run the simulation with the current potentials and return the trajectory.
        """
        raise NotImplementedError()

class LAMMPS(Engine):
    def __init__(self, env, lammps, template, args=None, procs=None, threads=None):
        super(LAMMPS, self).__init__(env)

        self.lammps = lammps
        self.template = template
        self.args = args if args is not None else ""

        self.procs = procs
        self.threads = threads

    def run(self, step, potential):
        # process the potentials into table files
        for i,j in potential:
            if j >= i:
                f = self.env.scratch('{s}/pair.{i}.{j}.dat'.format(s=step,i=i,j=j))
                print("Writing {}".format(f))

        # inject the tabulated pair potentials into the template
        file_ = self.template

        # run the simulation out of the scratch directory
        with utils.TemporaryWorkingDirectory(self.env.scratch()):
            cmd = "{lmp} -in {fn} -nocite {args}".format(lmp=self.lammps, fn=file_, args=self.args)
            self.env.call(cmd, procs=self.procs, threads=self.threads)

        # load in the simulated trajectory
        traj = self._load_trajectory(step)

        return traj

    def _load_trajectory(self, step):
        pass
