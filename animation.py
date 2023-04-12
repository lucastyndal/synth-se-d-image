# Python built-in modules
from bisect import bisect_left      # search sorted keyframe lists

# External, non built-in modules
import OpenGL.GL as GL              # standard Python OpenGL wrapper
import glfw                         # lean window system wrapper for OpenGL
import numpy as np                  # all matrix manipulations & OpenGL args

from core import Node
from transform import (lerp, quaternion_slerp, quaternion_matrix, translate,
                       scale, identity)


# -------------- Keyframing Utilities TP6 ------------------------------------
class KeyFrames:
    """ Stores keyframe pairs for any value type with interpolation_function"""
    def __init__(self, time_value_pairs, interpolation_function=lerp):
        if isinstance(time_value_pairs, dict):  # convert to list of pairs
            time_value_pairs = time_value_pairs.items()
        keyframes = sorted(((key[0], key[1]) for key in time_value_pairs))
        self.times, self.values = zip(*keyframes)  # pairs list -> 2 lists
        self.interpolate = interpolation_function

    def value(self, time):
        """ Computes interpolated value from keyframes, for a given time """

        # 1. ensure time is within bounds else return boundary keyframe
        if time <= self.times[0] :
            return self.values[0]
        elif time >= self.times[-1] :
            return self.values[-1]

        # 2. search for closest index entry in self.times, using bisect_left
        index = bisect_left(self.times, time)-1
        # 3. using the retrieved index, interpolate between the two neighboring
        # values in self.values, using the stored self.interpolate function
        return self.interpolate(self.values[index], self.values[index+1], (time-self.times[index])/(self.times[index+1]-self.times[index]))


class TransformKeyFrames:
    """ KeyFrames-like object dedicated to 3D transforms """
    def __init__(self, translate_keys, rotate_keys, scale_keys):
        self.T = KeyFrames(translate_keys, lerp)
        self.R = KeyFrames(rotate_keys, quaternion_slerp)
        self.S = KeyFrames(scale_keys, lerp)

    def value(self, time):
        """ Compute each component's interpolation and compose TRS matrix """
        TR = self.T.value(time)
        RO = quaternion_matrix(self.R.value(time))
        SC = self.S.value(time)

        return np.array([[RO[0][0]*SC,RO[0][1]*SC,RO[0][2]*SC,TR[0]],
                         [RO[1][0]*SC,RO[1][1]*SC,RO[1][2]*SC,TR[1]],
                         [RO[2][0]*SC,RO[2][1]*SC,RO[2][2]*SC,TR[2]],
                         [     0     ,      0    ,     0     ,  1  ]])


class KeyFrameControlNode(Node):
    """ Place node with transform keys above a controlled subtree """
    def __init__(self, trans_keys, rot_keys, scale_keys, transform=identity()):
        super().__init__(transform=transform)
        self.keyframes = TransformKeyFrames(trans_keys, rot_keys, scale_keys)

    def draw(self, primitives=GL.GL_TRIANGLES, **uniforms):
        """ When redraw requested, interpolate our node transform from keys """
        self.transform = self.keyframes.value(glfw.get_time()%10)
        super().draw(primitives=primitives, **uniforms)


# -------------- Linear Blend Skinning : TP7 ---------------------------------
class Skinned:
    """ Skinned mesh decorator, passes bone world transforms to shader """
    def __init__(self, mesh, bone_nodes, bone_offsets):
        self.mesh = mesh

        # store skinning data
        self.bone_nodes = bone_nodes
        self.bone_offsets = np.array(bone_offsets, np.float32)

    def draw(self, **uniforms):
        world_transforms = [node.world_transform for node in self.bone_nodes]
        uniforms['bone_matrix'] = world_transforms @ self.bone_offsets
        self.mesh.draw(**uniforms)
