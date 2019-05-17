import gzip

from . import core

class LAMMPS(core.TrajectoryReader):
    def __init__(self, filename, tags=0, coordinates=(2,3,4), types=1):
        super().__init__(filename)
        self._gzip = '.gz' in self.filename

        self.tags = tags
        self.types = types
        try:
            self.coordinates = tuple(coordinates)
            if len(self.coordinates) != 3:
                raise TypeError('LAMMPS coordinate indexes must be a 3-tuple.')
        except TypeError:
            print('LAMMPS coordinate indexes must be a 3-tuple.')
            raise

    def load(self, env, step):
        traj = Trajectory()

        with env.data(step):
            with open(self.filename, 'r') as f:
                state = 0
                snap = None
                step = None

                line = f.readline()
                while line:
                    # timestep line first
                    if state == 0 and self._label('ITEM: TIMESTEP') in line:
                        step = self._forceread(f,'Could not read LAMMPS timestep')
                        step = int(step)
                        state += 1

                    # number of particles second
                    if state == 1 and self._label('ITEM: NUMBER OF ATOMS') in line:
                        N = self._forceread(f,'Could not read LAMMPS number of particles.')
                        N = int(N)
                        snap = Snapshot(N)
                        state += 1

                    # box size third
                    if state == 2 and self._label('ITEM: BOX BOUNDS') in line:
                        if self._label('xy xz yz') in line:
                            raise IOError('LAMMPS box must be orthorhombic.')

                        periodic = line.strip()[-8:]
                        print(periodic)

                        box_x = self._forceread(f,'Could not read LAMMPS box x.')
                        box_y = self._forceread(f,'Could not read LAMMPS box y.')
                        box_z = self._forceread(f,'Could not read LAMMPS box z.')

                        box_x = [float(x) for x in box_x.split()]
                        box_y = [float(x) for x in box_y.split()]
                        box_z = [float(x) for x in box_z.split()]
                        if len(box_x) != 2 or len(box_y) != 2 or len(box_z) != 2:
                            raise IOError('Could not read LAMMPS box dimensions.')

                        snap.box = Box(L=tuple(zip(box_x,box_y,box_z)))
                        state += 1

                    # atoms come fourth
                    if state == 3 and self._label('ITEM: ATOMS') in line:
                        # read particles
                        for i in range(snap.particles.N):
                            atom = self._forceread(f,'Could not read LAMMPS atom.')
                            atom = atom.split()

                            tag = int(atom[self.tags]) - 1
                            snap.positions[tag] = [float(atom[c]) for c in self.coordinates]
                            snap.types[tag] = atom[self.types]

                        # wrap particles back into the box
                        snap.positions = snap.box.wrap(snap.positions)
                        state += 1

                    # snapshot is finished
                    if state == 4:
                        traj.append(snap)
                        state = 0
                        snap = None
                        step = None

                    line = f.readline()

                if state != 0:
                    raise IOError('Did not cleanly finish reading LAMMPS trajectory.')

        return traj

    def _label(self, label):
        if self._gzip:
            return label.encode()
        else:
            return label

    def _forceread(self, f, msg=''):
        try:
            line = f.readline()
        except IOError:
            print(msg)
            raise
        if not line:
            raise IOError(msg)

        return line
