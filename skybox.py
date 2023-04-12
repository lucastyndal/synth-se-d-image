import numpy as np
from core import Node, Mesh
from copy import deepcopy
import OpenGL.GL as GL              # standard Python OpenGL wrapper

from PIL import Image               # load texture maps
from core import Mesh

# -------------- OpenGL Texture Wrapper ---------------------------------------
class Texture:
    """ Helper class to create and automatically destroy textures """
    def __init__(self, tex_file, wrap_mode=GL.GL_REPEAT,
                 mag_filter=GL.GL_LINEAR, min_filter=GL.GL_LINEAR_MIPMAP_LINEAR,
                 tex_type=GL.GL_TEXTURE_2D):
        self.glid = GL.glGenTextures(1)
        self.type = tex_type
        try:
            # imports image as a numpy array in exactly right format
            tex = Image.open(tex_file).convert('RGBA')
            GL.glBindTexture(tex_type, self.glid)
            GL.glTexImage2D(tex_type, 0, GL.GL_RGBA, tex.width, tex.height,
                            0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, tex.tobytes())
            GL.glTexParameteri(tex_type, GL.GL_TEXTURE_WRAP_S, wrap_mode)
            GL.glTexParameteri(tex_type, GL.GL_TEXTURE_WRAP_T, wrap_mode)
            GL.glTexParameteri(tex_type, GL.GL_TEXTURE_MIN_FILTER, min_filter)
            GL.glTexParameteri(tex_type, GL.GL_TEXTURE_MAG_FILTER, mag_filter)
            GL.glGenerateMipmap(tex_type)
            print(f'Loaded texture {tex_file} ({tex.width}x{tex.height}'
                  f' wrap={str(wrap_mode).split()[0]}'
                  f' min={str(min_filter).split()[0]}'
                  f' mag={str(mag_filter).split()[0]})')
        except FileNotFoundError:
            print("ERROR: unable to load texture file %s" % tex_file)

    def __del__(self):  # delete GL texture from GPU when object dies
        GL.glDeleteTextures(self.glid)


# -------------- Textured mesh decorator --------------------------------------
class Textured:
    """ Drawable mesh decorator that activates and binds OpenGL textures """
    def __init__(self, drawable : Mesh, **textures):
        self.drawable = drawable
        self.textures = textures

    def draw(self, primitives=GL.GL_TRIANGLES, **uniforms):
        for index, (name, texture) in enumerate(self.textures.items()):
            GL.glActiveTexture(GL.GL_TEXTURE0 + index)
            GL.glBindTexture(texture.type, texture.glid)
            uniforms[name] = index
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        self.drawable.draw(primitives=primitives, **uniforms)
        GL.glDisable(GL.GL_BLEND)



# -------------------- SkyBox -------------------------
class SkyBoxFace(Textured):
    def __init__(self, shader, base_coords, indices, texture):

        #gl_Indices = [indices[0], indices[1], indices[2], indices[2], indices[3], indices[0]]
        gl_Indices = [0, 1, 2, 2, 3, 0]

        gl_Coords = []
        for i in indices:
            gl_Coords.append(base_coords[i])

        gl_Coords = np.array(gl_Coords, np.float32)
        gl_Indices = np.array(gl_Indices, np.float32)

        mesh = Mesh(shader, attributes= dict(position=gl_Coords, tex_coord = np.array([[0.0,0.0], [1.0,0.0], [1.0,1.0], [0.0,1.0]], np.float32)), index = gl_Indices)
        self.texture = deepcopy(texture)
        super().__init__(mesh, diffuse_map = self.texture)

class SkyBox(Node):
    def __init__(self, shader, textures):
        super().__init__()
        size = 100
        skyboxVertices = [
        #   Coordinates
        [-size, -size,  size],   #0        7--------6
        [size, -size,  size],    #1       /|       /|
        [size, -size, -size],    #2      4--------5 |
        [-size, -size, -size],   #3      | |      | |           Camera ->>
        [-size,  size,  size],   #4      | 3------|-2
        [size,  size,  size],    #5      |/       |/
        [size,  size, -size],    #6      0--------1
        [-size,  size, -size]    #7
        ]
        
        base_coords = np.array(skyboxVertices, np.float32)

        textures["skybox_right"] = Texture("texture/right.png", GL.GL_CLAMP_TO_EDGE, GL.GL_NEAREST, GL.GL_NEAREST)
        self.add(SkyBoxFace(shader, base_coords, [5,4,0,1], textures["skybox_right"]))

        textures["skybox_left"] = Texture("texture/left.png", GL.GL_CLAMP_TO_EDGE, GL.GL_NEAREST, GL.GL_NEAREST)
        self.add(SkyBoxFace(shader, base_coords, [7,6,2,3], textures["skybox_left"]))

        textures["skybox_top"] = Texture("texture/top.png", GL.GL_CLAMP_TO_EDGE, GL.GL_NEAREST, GL.GL_NEAREST)
        #self.add(SkyBoxFace(shader, base_coords, [4,5,6,7], textures["skybox_top"]))
        self.add(SkyBoxFace(shader, base_coords, [7,4,5,6], textures["skybox_top"]))

        textures["skybox_bottom"] = Texture("texture/bottom.png", GL.GL_CLAMP_TO_EDGE, GL.GL_NEAREST, GL.GL_NEAREST)
        self.add(SkyBoxFace(shader, base_coords, [2,1,0,3], textures["skybox_bottom"]))

        textures["skybox_back"] = Texture("texture/back.png", GL.GL_CLAMP_TO_EDGE, GL.GL_NEAREST, GL.GL_NEAREST)
        self.add(SkyBoxFace(shader, base_coords, [4,7,3,0], textures["skybox_back"]))

        textures["skybox_front"] = Texture("texture/front.png", GL.GL_CLAMP_TO_EDGE, GL.GL_NEAREST, GL.GL_NEAREST)
        self.add(SkyBoxFace(shader, base_coords, [6,5,1,2], textures["skybox_front"]))
