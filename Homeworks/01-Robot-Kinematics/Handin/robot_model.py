#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author  :   Arthals
# @File    :   robot_model.py
# @Time    :   2025/03/08 03:54:23
# @Contact :   zhuozhiyongde@126.com
# @Software:   Visual Studio Code


import os
from typing import List
import numpy as np
import xml.etree.ElementTree as ET

from urdf_types import Link, Joint, FixedJoint, RevoluteJoint
from config import RobotConfig
from rotation import rpy_to_mat
from utils import str_to_np
from vis import Vis


class RobotModel:
    robot_cfg: RobotConfig
    links: List[Link]
    joints: List[Joint]

    def __init__(self, robot_cfg: RobotConfig):
        """
        Initialize the RobotModel with the given RobotConfig

        Parameters
        ----------
        robot_cfg : RobotConfig
            The configuration of the robot
        """
        self.robot_cfg = robot_cfg
        self.load_urdf(robot_cfg)

    def fk(self, qpos: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics for all of the links.

        Here we assume the i-th joint has i-th link as parent
        and i+1-th link as child.

        In practice multiple joints can have a shared parent link.
        But dealing with those special cases is not required in the homework.

        The result is in robot frame, which means that the first link
        has (0, 0, 0) as translation and I as rotation matrix

        Here we assume each link's frame, except the first one,
        is as same as its parent joint's frame.

        See urdf_types.py for the definition of Link and Joint.

        Parameters
        ----------
        qpos: np.ndarray
            The current joint angles with shape (J,)
            (which means its length is the number of revolute joints)

        Returns
        -------
        np.ndarray
            The poses of links with shape (L, 4, 4)
            (which means its length is the number of links)

        Note
        ----
        The 4*4 pose matrix combines translation and rotation.

        Its format is:
            R  t
            0  1
        where R is rotation matrix and t is translation vector
        """
        L = len(self.links)
        poses = np.zeros((L, 4, 4))
        poses[0] = np.eye(4)
        q_idx = 0
        for i in range(len(self.joints)):
            joint = self.joints[i]
            if isinstance(joint, RevoluteJoint):
                theta = qpos[q_idx]
                q_idx += 1
                axis = joint.axis / np.linalg.norm(joint.axis)
                x, y, z = axis
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)
                K = np.array(
                    [
                        [0, -z, y],
                        [z, 0, -x],
                        [-y, x, 0],
                    ]
                )
                R_theta = (
                    cos_t * np.eye(3) + (1 - cos_t) * np.outer(axis, axis) + sin_t * K
                )
                R = joint.rot @ R_theta
                t = joint.trans
            else:
                R = joint.rot
                t = joint.trans
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            poses[i + 1] = poses[i] @ T
        return poses

    def load_urdf(self, robot_cfg: RobotConfig):
        """
        Load the URDF into this RobotModel

        Theoretically one can write a general code that load
        everything only from the URDF, but it will make the
        code too complex. Thus we read the joints' name and
        links' name from the RobotConfig instead.

        Parameters
        ----------
        robot_cfg : RobotConfig
            The configuration of the robot
        """
        self.links = [None for _ in robot_cfg.link_names]
        self.joints = [None for _ in robot_cfg.joint_names]
        tree = ET.parse(robot_cfg.urdf_path)
        root = tree.getroot()
        for child in root:
            if child.tag == "link":
                idx = robot_cfg.link_names.index(child.attrib["name"])
                self.links[idx] = Link(
                    name=child.attrib["name"],
                    visual_meshes=[
                        os.path.join(
                            os.path.dirname(robot_cfg.urdf_path), m.attrib["filename"]
                        )
                        for m in child.findall("./visual/geometry/mesh")
                    ],
                )
            elif child.tag == "joint":
                idx = robot_cfg.joint_names.index(child.attrib["name"])
                joint_type = child.attrib["type"]
                kwargs = dict(
                    name=child.attrib["name"],
                    trans=str_to_np(child.find("origin").attrib["xyz"]),
                    rot=rpy_to_mat(str_to_np(child.find("origin").attrib["rpy"])),
                )
                if joint_type == "fixed":
                    self.joints[idx] = FixedJoint(**kwargs)
                elif joint_type == "revolute":
                    self.joints[idx] = RevoluteJoint(
                        axis=str_to_np(child.find("axis").attrib["xyz"]),
                        lower_limit=float(child.find("limit").attrib["lower"]),
                        upper_limit=float(child.find("limit").attrib["upper"]),
                        **kwargs,
                    )

    def vis(self, poses: np.ndarray, color: str) -> list:
        """
        A helper function to visualize the fk result with plotly.

        You can modify it for debugging

        Parameters
        ----------
        poses: np.ndarray
            The poses of each link with shape (L, 4, 4)

        color: str (or any other format supported by plotly)
            The color of the meshes shown in visualization

        Returns
        -------
        A list of plotly objects that can be shown in Vis.show
        """
        vis_list = []
        for l, p in zip(self.links, poses):
            vis_list += Vis.pose(p[:3, 3], p[:3, :3])
            for m in l.visual_meshes:
                vis_list += Vis.mesh(path=m, trans=p[:3, 3], rot=p[:3, :3], color=color)
        return vis_list


if __name__ == "__main__":
    # a simple test to check if the code is working
    # you can modify it to test your code
    from config import get_robot_config

    cfg = get_robot_config("galbot")
    robot_model = RobotModel(cfg)

    gt = np.load(os.path.join("data", "fk.npz"))
    idx = 0
    q = gt["q"][idx]
    gt_poses = gt["poses"][idx]

    # the correct answer is the green one
    gt_vis = robot_model.vis(gt_poses, color="lightgreen")

    my_poses = robot_model.fk(q * 0)
    # your answer is the brown one
    my_vis = robot_model.vis(my_poses, color="brown")

    # it will be shown in the browser
    # if not, you can input a html path to this function and open it manually
    # if a new page is shown but nothing is displayed, you can try refreshing the page
    Vis.show(gt_vis + my_vis, path=None)
