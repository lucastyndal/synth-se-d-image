from core import Mesh
from texture import Textured

from random import random

import numpy as np

# The Perlin Noise basic step
def noisestep(buf, x : int, y : int, N : int):
    if N > 1:
        # Set the center with a small random offset
        buf[x+N//2][y+N//2] = np.average([buf[x][y], buf[x][y+N//2], buf[x+N//2][y+N//2], buf[x+N//2][y]]) + random()
        # Set the borders with 3 mid average function
        buf[x][y+N//2]   = np.average([buf[x+N//2][y+N//2], buf[x][y],   buf[x][y+N]])
        buf[x+N][y+N//2] = np.average([buf[x+N//2][y+N//2], buf[x+N][y], buf[x+N][y+N]])
        buf[x+N//2][y]   = np.average([buf[x+N//2][y+N//2], buf[x][y], buf[x+N][y]])
        buf[x+N//2][y+N] = np.average([buf[x+N//2][y+N//2], buf[x][y+N], buf[x+N][y+N]])
        # Recursive call
        noisestep(buf, x, y, N//2)
        noisestep(buf, x+N//2, y, N//2)
        noisestep(buf, x+N//2, y+N//2, N//2)
        noisestep(buf, x, y+N//2, N//2)

# ------------------- The ground ----------------------
class Ground(Textured):
    """ Simple first textured object """
    def __init__(self, shader, texture, size):
        # Make some noise on a 2*size sized plane
        # Initialize the noise buffer
        self.shader = shader
        
        x=size
        y=size
        f=3
        a=0.5
        
        sub_div=400
        Z=np.empty([sub_div+1,sub_div+1])
        
        X=np.empty(sub_div+1)
        Y=np.empty(sub_div+1)
        pas=-x/2
        S=np.empty([sub_div*sub_div,3])  #list sommet
        for i in range(len(X)):
            pas=pas+x/sub_div
            X[i]=pas
            Y[i]=pas
        for i in range(len(X)):
            for j in range(len(Y)):
                Z[i][j]=(a*np.cos(f*x/sub_div*i)*np.sin(f*y/sub_div*j))**2
        pas=x/sub_div
        for i in range(sub_div):
            for j in range(sub_div):
                p=(X[i],Z[i][j]+1,Y[j])
                S[i*sub_div+j]=p

        position=np.array(S, 'f')
        #print(S)
        L=[]
        for i in range(sub_div-1):
            for j in range(sub_div-1):  #creation d'un carre on dit quel sommet appartient au carré
                L2=[]                          #premier triangle composant du carré 
                L2.append(i*sub_div+j)
                L2.append(i*sub_div+j+1)
                L2.append((i+1)*sub_div+j)
                L.append(L2)
                
                s1=S[L2[0]]                      #normal au triangle
                s2=S[L2[1]]
                s3=S[L2[2]]
 
                L2=[]                               #second triangle
                L2.append(i*sub_div+j+1)
                L2.append((i+1)*sub_div+j+1)
                L2.append((i+1)*sub_div+j)
                L.append(L2)
                s1=S[L2[0]]                      #normal au triangle
                s2=S[L2[1]]
                s3=S[L2[2]]
        self.index=np.array(L)
        
        tex_coords = [(i % 2, j % 2) for i in range(-size, size+1) for j in range(-size, size+1)]
        
        
        mesh = Mesh(shader, attributes=dict(position=position, tex_coord=tex_coords), index=self.index)

        # setup & upload texture to GPU, bind it to shader name 'diffuse_map'
        self.texture = texture
        super().__init__(mesh, diffuse_map = self.texture)