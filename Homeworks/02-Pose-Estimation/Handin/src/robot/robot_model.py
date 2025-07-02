import os
import yaml
from typing import Optional, List, Dict, Tuple
from transforms3d.euler import euler2mat
import numpy as np
import xml.etree.ElementTree as ET
import roboticstoolbox as rtb

from src.robot.cfg import RobotCfg
from src.utils import to_pose, rot_dist
from src.vis import Vis


class RobotModel:
    def __init__(self, robot_cfg: RobotCfg):
        """
        Initialize the RobotModel with the given configuration.

        Parameters
        ----------
        robot_cfg : RobotCfg
            Configuration object containing robot parameters.
        """
        self.cfg = robot_cfg
        self.robot = rtb.ERobot.URDF(os.path.abspath(robot_cfg.path_urdf))
        joints = [l for l in self.robot.links if l.isjoint]
        self.joint_lower_limit = np.array([j.qlim[0] for j in joints])
        self.joint_upper_limit = np.array([j.qlim[1] for j in joints])
        self.setup_visual()
        self.setup_collision()

    def setup_visual(self):
        """Load the visual mesh information"""
        self.link_meshes = dict()
        tree = ET.parse(self.cfg.path_urdf)
        root = tree.getroot()
        for child in root:
            if child.tag == "link":
                link_name = child.attrib["name"]
                self.link_meshes[link_name] = []
                for visual in child.findall("./visual"):
                    origin = visual.find("origin")
                    if origin is not None:
                        origin_xyz = [
                            float(coord)
                            for coord in origin.attrib.get("xyz", "0 0 0").split()
                        ]
                        origin_rpy = [
                            float(angle)
                            for angle in origin.attrib.get("rpy", "0 0 0").split()
                        ]
                    else:
                        origin_xyz = [0, 0, 0]
                        origin_rpy = [0, 0, 0]
                    pose = to_pose(
                        trans=np.array(origin_xyz), rot=euler2mat(*origin_rpy)
                    )

                    for geo in visual.findall("./geometry"):
                        for m in geo.findall("./mesh"):
                            if m.tag == "mesh":
                                mesh_path = os.path.join(
                                    os.path.dirname(self.cfg.path_urdf),
                                    m.attrib["filename"],
                                )
                                self.link_meshes[link_name].append(
                                    dict(
                                        path=mesh_path,
                                        pose=pose.copy(),
                                    )
                                )
                            else:
                                raise NotImplementedError(
                                    f"Mesh type {m.tag} not supported."
                                )

    def setup_collision(self):
        """Load the collision sphere information"""
        self.collision_spheres = dict()
        with open(self.cfg.path_collision, "r") as f:
            collision_data = yaml.safe_load(f)["collision_spheres"]
        for link_name, spheres in collision_data.items():
            centers = np.array([s["center"] for s in spheres])
            radii = np.array([s["radius"] for s in spheres])
            self.collision_spheres[link_name] = dict(center=centers, radius=radii)

    def fk_link(
        self, qpos: np.ndarray, link_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the forward kinematics of a specific link.

        Parameters
        ----------
        qpos : np.ndarray
            Joint positions of the robot.
        link_name : str
            The name of the link for which to compute the forward kinematics.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The translation (3,) and rotation matrix (3, 3) of the specified link.
        """
        fk = self.robot.fkine(qpos, end=link_name)
        return fk.A[:3, 3], fk.A[:3, :3]

    def fk_eef(self, qpos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the forward kinematics of the end-effector (EEF) defined in the RobotCfg.

        Parameters
        ----------
        qpos : np.ndarray
            Joint positions of the robot.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The translation (3,) and rotation matrix (3, 3) of the end-effector.
        """
        return self.fk_link(qpos, self.cfg.link_eef)

    def fk_camera(self, qpos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the forward kinematics of the camera defined in the RobotCfg.

        Parameters
        ----------
        qpos : np.ndarray
            Joint positions of the robot.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The translation (3,) and rotation matrix (3, 3) of the camera.
        """
        link_trans, link_rot = self.fk_link(qpos, self.cfg.camera_cfg.link_name)
        rel_cam_trans, rel_cam_rot = (
            self.cfg.camera_cfg.extrinsics[:3, 3],
            self.cfg.camera_cfg.extrinsics[:3, :3],
        )
        cam_trans = link_trans + link_rot @ rel_cam_trans
        cam_rot = link_rot @ rel_cam_rot
        return cam_trans, cam_rot

    def fk_all_link(self, qpos: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute the forward kinematics of all links in the robot.

        Parameters
        ----------
        qpos : np.ndarray
            Joint positions of the robot.

        Returns
        -------
        Dict[str, Tuple[np.ndarray, np.ndarray]]
            A dictionary mapping link names to their translation (3,) and rotation matrix (3, 3).
        """
        ret = dict()
        for link in self.robot.links:
            fk = self.robot.fkine(qpos, end=link.name)
            ret[link.name] = fk.A[:3, 3], fk.A[:3, :3]
        return ret

    def uniform_rand_qpos(self) -> np.ndarray:
        """
        Randomly sample a joint position within the robot's joint limits.

        Returns
        -------
        np.ndarray
            A randomly sampled joint position vector.
        """
        qpos = np.random.uniform(self.joint_lower_limit, self.joint_upper_limit)
        return qpos

    def ik(
        self,
        trans: np.ndarray,
        rot: np.ndarray,
        init_qpos: Optional[np.ndarray] = None,
        retry_times: int = 10,
        trans_tol=1e-3,
        rot_tol=1e-2,
        delta_thresh: float = None,
    ) -> Tuple[bool, np.ndarray]:
        """
        Inverse kinematics solver for the robot.

        Parameters
        ----------
        trans : np.ndarray (3,)
            The desired translation of the end-effector.
        rot : np.ndarray (3, 3)
            The desired rotation of the end-effector.
        init_qpos : Optional[np.ndarray], optional (J,)
            Initial joint positions for the solver (default is None).
        retry_times : int, optional
            Number of retries for the solver (default is 10).
        trans_tol : float, optional
            Tolerance for translation (default is 1e-3).
        rot_tol : float, optional
            Tolerance for rotation (default is 1e-2).
        delta_thresh : float, optional
            Threshold for joint position change when init_qpos is provided.

            This is used to constrain the solution to be close to the initial qpos.

        Returns
        -------
        Tuple[bool, np.ndarray]
            A tuple indicating success and the resulting joint positions.

        Notes
        -----
        This function only return the joint angles that affect the end-effector.
        """
        pose = to_pose(trans, rot)
        for _ in range(retry_times):
            ik_result, success, _, _, _ = self.robot.ik_lm_chan(
                pose, end=self.cfg.link_eef, q0=init_qpos
            )
            if success:
                if delta_thresh is not None:
                    if np.linalg.norm(ik_result - init_qpos) > delta_thresh:
                        continue
                t, r = self.fk_link(ik_result, self.cfg.link_eef)
                if np.linalg.norm(t - trans) < trans_tol and rot_dist(r, rot) < rot_tol:
                    return True, ik_result
        return False, ik_result

    def vis(
        self,
        qpos: np.ndarray,
        mode: str = "visual",
        opacity: float = 1.0,
        color: str = "purple",
    ) -> list:
        """
        Return the visualization meshes or collision spheres for the robot.

        Can be used with plotly visualization functions in src.vis

        Parameters
        ----------
        qpos : np.ndarray
            Joint positions of the robot.
        mode : str, optional
            Visualization mode, either 'visual' or 'collision' (default is 'visual').
        opacity : float, optional
            Opacity of the meshes or spheres (default is 1.0).
        color : str, optional
            Color of the meshes or spheres (default is 'purple').
        """
        lst = []
        fk_links = self.fk_all_link(qpos)
        if mode == "visual":
            for link in self.link_meshes:
                trans, rot = fk_links[link]
                link_pose = to_pose(trans, rot)
                for m in self.link_meshes[link]:
                    mesh_path, mesh_pose = m["path"], m["pose"]
                    mesh_pose = link_pose @ mesh_pose
                    lst += Vis.mesh(
                        mesh_path,
                        trans=mesh_pose[:3, 3],
                        rot=mesh_pose[:3, :3],
                        opacity=opacity,
                        color=color,
                    )
        elif mode == "collision":
            for link, spheres in self.collision_spheres.items():
                trans, rot = fk_links[link]
                for center, radius in zip(spheres["center"], spheres["radius"]):
                    lst += Vis.sphere(
                        center=trans + rot @ center,
                        radius=radius,
                        opacity=opacity,
                        color=color,
                    )
        return lst

    def check_collision(
        self, qpos: np.ndarray, pc: np.ndarray, thresh: float = 0.0025
    ) -> Tuple[bool, str]:
        """
        Check if the point cloud collides with the robot's collision spheres or between two links.

        Parameters
        ----------
        qpos : np.ndarray (J,)
            Joint positions of the robot.
        pc : np.ndarray (N, 3)
            Point cloud data to check for collisions.
        thresh : float, optional
            Collision threshold (default is 0.0025).

        Returns
        -------
        Tuple[bool, str]
            A tuple indicating if a collision occurred and the cause of the collision.
        """
        fk_links = self.fk_all_link(qpos)
        link_spheres = dict()
        for link, spheres in self.collision_spheres.items():
            trans, rot = fk_links[link]
            center = np.einsum("ab,nb->na", rot, spheres["center"]) + trans
            link_spheres[link] = dict(center=center, radius=spheres["radius"])

        for link, spheres in link_spheres.items():
            for center, radius in zip(spheres["center"], spheres["radius"]):
                dist = np.linalg.norm(pc[:, None] - center[None], axis=-1) - radius
                if np.any(dist < thresh):
                    return True, link

        link_list = list(link_spheres.keys())
        for i in range(len(link_list)):
            for j in range(i + 1, len(link_list)):
                link_a, link_b = link_list[i], link_list[j]
                if self.cfg.is_collision_ignore(link_a, link_b):
                    continue
                for center_a, radius_a in zip(
                    link_spheres[link_a]["center"], link_spheres[link_a]["radius"]
                ):
                    for center_b, radius_b in zip(
                        link_spheres[link_b]["center"], link_spheres[link_b]["radius"]
                    ):
                        dist = np.linalg.norm(center_a - center_b) - (
                            radius_a + radius_b
                        )
                        if dist < thresh:
                            return True, f"{link_a} and {link_b}"

        return False, ""
