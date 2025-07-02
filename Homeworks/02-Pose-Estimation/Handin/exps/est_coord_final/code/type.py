from typing import List
from dataclasses import dataclass, field
import numpy as np
from .vis import Vis


@dataclass
class Grasp:
    trans: np.ndarray
    """(3,) translation vector"""
    rot: np.ndarray
    """(3, 3) rotation matrix"""
    width: float
    """how much the gripper should open"""

    def vis(
        self, finger_width=0.005, tail_length=0.05, depth=0.05, opacity=None, color=None
    ) -> list:
        """
        Visualize the grasp using boxes in plotly

        Can be used similar to other functions in src.vis, return a list of plotly objects
        """
        centers = np.array(
            [
                [-depth / 2, (self.width + finger_width) / 2, 0],
                [-depth / 2, -(self.width + finger_width) / 2, 0],
                [-depth - finger_width / 2, 0, 0],
                [-depth - finger_width - tail_length / 2, 0, 0],
            ]
        )
        scales = np.array(
            [
                [depth, finger_width, finger_width],
                [depth, finger_width, finger_width],
                [finger_width, self.width + 2 * finger_width, finger_width],
                [tail_length, finger_width, finger_width],
            ]
        )
        centers = np.einsum("ij,kj->ki", self.rot, centers) + self.trans
        box_plotly_list = []
        for i in range(4):
            box_plotly_list += Vis.box(scales[i], centers[i], self.rot, opacity, color)
        return box_plotly_list


@dataclass
class Obj:
    # general
    name: str = None
    """Name of the object"""
    pose: np.ndarray = field(default_factory=lambda: np.eye(4))
    """(4, 4) pose matrix of the object in world frame"""

    # for mujoco
    density: float = 1000
    friction: str = field(default_factory=lambda: "1.0 0.001 0.0005")
    mass: float = None
    fixed_body: bool = False
    visual_only: bool = False
    color: np.ndarray = field(default_factory=lambda: np.array((0.7, 0.7, 0.7, 1.0)))

    @property
    def trans(self) -> np.ndarray:
        return self.pose[:3, 3]

    @property
    def rot(self) -> np.ndarray:
        return self.pose[:3, :3]

    @property
    def geom_id(self) -> str:
        raise NotImplementedError


@dataclass
class Box(Obj):
    size: np.ndarray = field(default_factory=lambda: np.ones(3))
    """The size of the box with shape (3,)"""

    @property
    def geom_id(self) -> str:
        return f"Box_{self.size[0]:.9f}_{self.size[1]:.9f}_{self.size[2]:.9f}"


@dataclass
class Mesh(Obj):
    path: str = None
    """The path to the mesh file"""
    convex_decompose_paths: List[str] = None
    """A list of paths to the convex decomposed meshes"""
    scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    """Scale the mesh with this array, shape (3,)"""

    @property
    def geom_id(self) -> str:
        return f'Mesh_{self.path.replace("/", "_")}_{self.scale[0]:.9f}_{self.scale[1]:.9f}_{self.scale[2]:.9f}'


@dataclass
class Scene:
    obj_list: List[Obj] = field(default_factory=list)

    def get_obj_by_name(self, name: str) -> Obj:
        obj_name_list = [obj.name for obj in self.obj_list]
        assert len(obj_name_list) == len(
            set(obj_name_list)
        ), "Object names must be unique."
        assert name in obj_name_list, f"Object {name} not found in the scene."
        return self.obj_list[obj_name_list.index(name)]
