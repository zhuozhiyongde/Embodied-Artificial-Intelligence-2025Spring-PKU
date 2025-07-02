debug_str = """
    <default class="visual">
        <geom type="mesh" contype="1" conaffinity="0" group="2"/>
    </default>
    <default class="collision">
        <geom type="mesh" contype="1" conaffinity="0" group="3"/>
    </default>
"""


DEFAULT_COMBINED_MJSCENE = """
<mujoco model="default_scene">
    <compiler angle="radian" autolimits="true"/>
    <option cone="elliptic" impratio="30"/>

    <default>
        <general ctrllimited="true"/>
    </default>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20" offwidth="1280" offheight="720"/>
  </visual>

  <asset>
    <material name="debug_red" rgba="1 0 0 0.5"/>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <geom name="floor" size="0 0 0.01" type="plane" material="groundplane" priority="1" friction="0.6" condim="3" contype="1" conaffinity="0"/>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <body name="debug_axis" pos="0 0 -1" mocap="true">
        <geom type="cylinder" fromto="0 0 0 0.1 0 0" size="0.005" rgba="1 0 0 0.2" contype="2" conaffinity="0" group="2"/>
        <geom type="cylinder" fromto="0 0 0 0 0.1 0" size="0.005" rgba="0 1 0 0.2" contype="2" conaffinity="0" group="2"/>
        <geom type="cylinder" fromto="0 0 0 0 0 0.1" size="0.005" rgba="0 0 1 0.2" contype="2" conaffinity="0" group="2"/>
    </body>
    <body name="camera_body_head" pos="0 0 1" mocap="true">
      <camera name="moving_cam_head" mode="fixed" pos="0 0 0"/>
    </body>
    <body name="camera_body_wrist" pos="0 0 1" mocap="true">
      <camera name="moving_cam_wrist" mode="fixed" pos="0 0 0"/>
    </body>
  </worldbody>
  

</mujoco>
"""

DEFAULT_GROUD_GEOM = dict(
    name="ground", size=[0, 0, 0.05], type="plane", material="groundplane"
)

DEBUG_AXIS_BODY_NAME = "debug_axis"
