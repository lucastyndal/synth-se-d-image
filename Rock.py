from core import Node, load
from core import Viewer, Node, Mesh

import OpenGL.GL as GL              # standard Python OpenGL wrapper
import glfw                         # lean window system wrapper for OpenGL
from random import random
import numpy as np
from transform import translate, identity, rotate, scale



class Rock(Node):
    def __init__(self, shader ):
        super().__init__()
        self.add(*load("suzanne.obj", shader))
        # Alive ?
        self.a=1
        self.alive = False
    
    def summon(self):
        # Make it alive !
        self.alive = True
        v=random()*25+10
        # Generate random values
        angle = (45 + random() * 30) * (np.pi/180)
        angle2=random() *2*np.pi
        self.pos=np.array([0,0,0])
        self.speed=np.array([np.cos(angle)*np.cos(angle2)*v,np.sin(angle)*v,np.cos(angle)*np.sin(angle2)*v])

    def key_handler(self, key, action):
        if action == glfw.PRESS:
            if key == glfw.KEY_U and not (self.alive):
                self.summon()
                print("appui U")
    
    def draw(self, primitives=GL.GL_TRIANGLES, **uniforms):
        # IF THE ROCK IS ACTIVE
        self.a+=1
        if self.alive:
            # frictions
            n_pos=np.array([self.pos[0]+self.speed[0]*Viewer.dtime,self.pos[1]+self.speed[1]*Viewer.dtime,self.pos[2]+self.speed[2]*Viewer.dtime])
            n_speed=np.array([self.speed[0],self.speed[1]-10*Viewer.dtime,self.speed[2]])
            print(n_speed[1])
            if(n_pos[1]<=-1):
                    self.alive=False
                    self.transform=translate(np.array([0,0,0]))
                    ##tanslate orginal pos
                    ##
                    ##
            else:
                self.transform=translate(self.pos)
                self.pos=n_pos
                self.speed=n_speed
                super().draw(primitives=primitives, **uniforms)

        else:
            super().draw(primitives=primitives, **uniforms)
