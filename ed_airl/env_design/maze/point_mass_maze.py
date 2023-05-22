from contextlib import contextmanager
import tempfile
import numpy as np

"""
from
https://github.com/justinjfu/inverse_rl/blob/9609933389459a3a54f5c01d652114ada90fa1b3/inverse_rl/envs/dynamic_mjc/mjc_models.py
"""

RIGHT = 1
LEFT = 0

class MJCTreeNode(object):
    def __init__(self, name):
        self.name = name
        self.attrs = {}
        self.children = []

    def add_attr(self, key, value):
        if isinstance(value, str):
            pass
        elif isinstance(value, list) or isinstance(value, np.ndarray):
            value = ' '.join([str(val).lower() for val in value])
        else:
            value = str(value).lower()

        self.attrs[key] = value
        return self

    def __getattr__(self, name):
        def wrapper(**kwargs):
            newnode =  MJCTreeNode(name)
            for (k, v) in kwargs.items():
                newnode.add_attr(k, v)
            self.children.append(newnode)
            return newnode
        return wrapper

    def dfs(self):
        yield self
        if self.children:
            for child in self.children:
                for node in child.dfs():
                    yield node

    def find_attr(self, attr, value):
        """ Run DFS to find a matching attr """
        if attr in self.attrs and self.attrs[attr] == value:
            return self
        for child in self.children:
            res = child.find_attr(attr, value)
            if res is not None:
                return res
        return None


    def write(self, ostream, tabs=0):
        contents = ' '.join(['%s="%s"'%(k,v) for (k,v) in self.attrs.items()])
        if self.children:
            ostream.write('\t'*tabs)
            ostream.write('<%s %s>\n' % (self.name, contents))
            for child in self.children:
                child.write(ostream, tabs=tabs+1)
            ostream.write('\t'*tabs)
            ostream.write('</%s>\n' % self.name)
        else:
            ostream.write('\t'*tabs)
            ostream.write('<%s %s/>\n' % (self.name, contents))

    def __str__(self):
        s = "<"+self.name
        s += ' '.join(['%s="%s"'%(k,v) for (k,v) in self.attrs.items()])
        return s+">"

class MJCModel(object):
    def __init__(self, name):
        self.name = name
        self.root = MJCTreeNode("mujoco").add_attr('model', name)

    @contextmanager
    def asfile(self):
        """
        Usage:
        model = MJCModel('reacher')
        with model.asfile() as f:
            print f.read()  # prints a dump of the model
        """
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.xml', delete=True) as f:
            self.root.write(f)
            f.seek(0)
            yield f

    def open(self):
        self.file = tempfile.NamedTemporaryFile(mode='w+', suffix='.xml', delete=True)
        self.root.write(self.file)
        self.file.seek(0)
        return self.file

    def close(self):
        self.file.close()

    def find_attr(self, attr, value):
        return self.root.find_attr(attr, value)

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        pass

def point_mass_maze(
        length=0.6,
        borders=True,

        wall_1=True,
        wall_1_x1=0,
        wall_1_y1=0,
        wall_1_x2=0,
        wall_1_y2=0,
        
        wall_2=True,
        wall_2_x1=0,
        wall_2_y1=0,
        wall_2_x2=0,
        wall_2_y2=0,
        
        wall_3=True,
        wall_3_x1=0,
        wall_3_y1=0,
        wall_3_x2=0,
        wall_3_y2=0,
        
        wall_4=False,
        wall_4_x1=0,
        wall_4_y1=0,
        wall_4_x2=0,
        wall_4_y2=0,
        
):
    mjcmodel = MJCModel('twod_maze')
    mjcmodel.root.compiler(inertiafromgeom="true", angle="radian", coordinate="local")
    mjcmodel.root.option(timestep="0.01", gravity="0 0 0", iterations="20", integrator="Euler")
    default = mjcmodel.root.default()
    default.joint(damping=1, limited='false')
    default.geom(friction=".5 .1 .1", density="1000", margin="0.002", condim="1", contype="2", conaffinity="1")

    worldbody = mjcmodel.root.worldbody()

    particle = worldbody.body(name='particle', pos=[length/2,0,0])
    particle.geom(name='particle_geom', type='sphere', size='0.03', rgba='0.0 0.0 1.0 1', contype=1)
    particle.site(name='particle_site', pos=[0,0,0], size=0.01)
    particle.joint(name='ball_x', type='slide', pos=[0,0,0], axis=[1,0,0])
    particle.joint(name='ball_y', type='slide', pos=[0,0,0], axis=[0,1,0])

    positions = [
        [0., length/2, 0.01],
        [length - 0.1, length/2, 0.01],
        [length/2, length-0.1, 0.01]
    ]
    for i in range(len(positions)):
        target = worldbody.body(name='target_%d' % i, pos=positions[i])
        target.geom(name='target_geom_%d' % i, conaffinity=2, type='sphere', size=0.02, rgba=[0,0.9,0.1,1])

    L = -0.1
    R = length
    U = length
    D = -0.1

    if borders:
        worldbody.geom(conaffinity=1, fromto=[L, D, .01, R, D, .01], name="sideS", rgba="0.9 0.4 0.6 1", size=".02", type="capsule")
        worldbody.geom(conaffinity=1, fromto=[R, D, .01, R, U, .01], name="sideE", rgba="0.9 0.4 0.6 1", size=".02", type="capsule")
        worldbody.geom(conaffinity=1, fromto=[L, U, .01, R, U, .01], name="sideN", rgba="0.9 0.4 0.6 1", size=".02", type="capsule")
        worldbody.geom(conaffinity=1, fromto=[L, D, .01, L, U, .01], name="sideW", rgba="0.9 0.4 0.6 1", size=".02", type="capsule")
        
    # walls
    if wall_1:
        worldbody.geom(conaffinity=1, fromto=[wall_1_x1, wall_1_y1, .01, wall_1_x2, wall_1_y2, .01], name="barrier1",
                       rgba="0.9 0.4 0.6 1", size=".02", type="capsule")
    if wall_2:
        worldbody.geom(conaffinity=1, fromto=[wall_2_x1, wall_2_y1, .01, wall_2_x2, wall_2_y2, .01], name="barrier2",
                       rgba="0.9 0.4 0.6 1", size=".02", type="capsule")
    if wall_3:
        worldbody.geom(conaffinity=1, fromto=[wall_3_x1, wall_3_y1, .01, wall_3_x2, wall_3_y2, .01], name="barrier3",
                       rgba="0.9 0.4 0.6 1", size=".02", type="capsule")
    if wall_4:
        worldbody.geom(conaffinity=1, fromto=[wall_4_x1, wall_4_y1, .01, wall_4_x2, wall_4_y2, .01], name="barrier4",
                       rgba="0.9 0.4 0.6 1", size=".02", type="capsule")
        
    actuator = mjcmodel.root.actuator()
    actuator.motor(joint="ball_x", ctrlrange=[-1.0, 1.0], ctrllimited=True)
    actuator.motor(joint="ball_y", ctrlrange=[-1.0, 1.0], ctrllimited=True)

    return mjcmodel