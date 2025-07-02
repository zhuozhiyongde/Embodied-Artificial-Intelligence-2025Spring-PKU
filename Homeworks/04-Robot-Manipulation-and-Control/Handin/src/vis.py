import os
from typing import Optional, Union
import trimesh as tm
import numpy as np
import plotly.graph_objects as go


def generate_uv_sphere(segments=16, rings=8):
    vertices = []
    faces = []

    # Generate vertices
    for i in range(rings + 1):
        v = i / rings
        phi = v * np.pi

        for j in range(segments + 1):
            u = j / segments
            theta = u * 2 * np.pi

            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)

            vertices.append([x, y, z])

    # Generate faces
    for i in range(rings):
        for j in range(segments):
            a = i * (segments + 1) + j
            b = a + segments + 1
            d = a + 1
            c = b + 1

            faces.append([a, b, c])
            faces.append([a, c, d])

    return np.array(vertices), np.array(faces)


class Vis:
    """ """

    def __init__(self):
        pass

    @staticmethod
    def box(
        scale: np.ndarray,  # (3, )
        trans: Optional[np.ndarray] = None,  # (3, )
        rot: Optional[np.ndarray] = None,  # (3, 3)
        opacity: Optional[float] = None,
        color: Optional[str] = None,
    ) -> list:
        """
        Visualize a box with given scale, translation and rotation

        Parameters
        ----------
        scale: np.ndarray
            The scale of the box with shape (3, )
        trans: Optional[np.ndarray]
            The translation of the box with shape (3, )
        rot: Optional[np.ndarray]
            The rotation of the box with shape (3, 3)
        opacity: Optional[float]
            The opacity of the box
        color: Optional[str]
            The color of the box

        Returns
        -------
        A list of plotly objects that can be shown in Vis.show
        """

        color = "violet" if color is None else color
        opacity = 1.0 if opacity is None else opacity
        trans = np.zeros(3) if trans is None else trans
        rot = np.eye(3) if rot is None else rot

        # 8 vertices of a cube
        corner = (
            np.array(
                [
                    [0, 0, 1, 1, 0, 0, 1, 1],
                    [0, 1, 1, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 1, 1, 1, 1],
                ]
            ).T
            - 0.5
        )
        corner *= scale
        corner = np.einsum("ij,kj->ki", rot, corner) + trans

        return [
            go.Mesh3d(
                x=corner[:, 0],
                y=corner[:, 1],
                z=corner[:, 2],
                i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                color=color,
                opacity=opacity,
            )
        ]

    @staticmethod
    def sphere(
        center: Optional[np.ndarray] = None,
        radius: Optional[float] = None,
        opacity: Optional[float] = None,
        color: Optional[str] = None,
    ) -> list:
        """
        Visualize a sphere with given center, radius, opacity and color

        Parameters
        ----------
        center: Optional[np.ndarray]
            The center of the sphere with shape (3, )
        radius: Optional[float]
            The radius of the sphere
        opacity: Optional[float]
            The opacity of the sphere
        color: Optional[str]
            The color of the sphere

        Returns
        -------
        A list of plotly objects that can be shown in Vis.show
        """
        center = np.zeros(3) if center is None else center
        radius = 1.0 if radius is None else radius
        color = "blue" if color is None else color
        opacity = 1.0 if opacity is None else opacity

        vertices, faces = generate_uv_sphere()
        vertices = center + vertices * radius
        return Vis.mesh(vertices=vertices, faces=faces, opacity=opacity, color=color)

    @staticmethod
    def pose(
        trans: np.ndarray,  # (3, )
        rot: np.ndarray,  # (3, 3)
        width: int = 7,
        length: float = 0.2,
    ) -> list:
        """
        Visualize the pose with red, green, blue lines

        Parameters
        ----------
        trans: np.ndarray
            The translation part of the pose with shape (3, )
        rot: np.ndarray
            The rotation part of the pose with shape (3, 3)
        width: int
            The width of the lines
        length: float
            The length of the lines

        Returns
        -------
        A list of plotly objects that can be shown in Vis.show
        """
        result = []
        for i, color in zip(range(3), ["red", "green", "blue"]):
            result += Vis.line(
                trans, trans + rot[:, i] * length, width=width, color=color
            )
        return result

    @staticmethod
    def line(
        p1: np.ndarray,  # (3)
        p2: np.ndarray,  # (3)
        width: int = None,
        color: str = None,
    ) -> list:
        """
        Visualize the line between two points

        Parameters
        ----------
        p1: np.ndarray
            The first point with shape (3, )
        p2: np.ndarray
            The second point with shape (3, )
        width: int
            The width of the lines
        length: float
            The length of the lines

        Returns
        -------
        A list of plotly objects that can be shown in Vis.show
        """
        color = "green" if color is None else color
        width = 1 if width is None else width

        pc = np.stack([p1, p2])
        x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
        return [
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line=dict(width=width, color=color),
            )
        ]

    @staticmethod
    def mesh(
        path: str = None,
        scale: float = 1.0,
        trans: np.ndarray = None,  # (3, )
        rot: np.ndarray = None,  # (3, 3)
        opacity: float = 1.0,
        color: str = "orange",
        vertices: Optional[np.ndarray] = None,  # (n, 3)
        faces: Optional[np.ndarray] = None,  # (m, 3)
    ) -> list:
        """
        Visualize the mesh with given path or given vertices and faces

        Parameters
        ----------
        path: str
            The path of the mesh file
        scale: float
            The scale of the mesh, default to be 1 (not change)
        trans: np.ndarray
            The translation of the mesh with shape (3, )
        rot: np.ndarray
            The rotation of the mesh with shape (3, 3)
        opacity: float
            The opacity of the mesh
        color: str
            The color of the mesh
        vertices: Optional[np.ndarray]
            The vertices of the mesh with shape (n, 3)
        faces: Optional[np.ndarray]
            The faces of the mesh with shape (m, 3)

        Returns
        -------
        A list of plotly objects that can be shown in Vis.show
        """

        trans = np.zeros(3) if trans is None else trans
        rot = np.eye(3) if rot is None else rot

        if path is not None:
            mesh = tm.load_mesh(path)
            vertices, faces = mesh.vertices, mesh.faces

        v = np.einsum("ij,kj->ki", rot, vertices * scale) + trans
        f = faces
        mesh_plotly = go.Mesh3d(
            x=v[:, 0],
            y=v[:, 1],
            z=v[:, 2],
            i=f[:, 0],
            j=f[:, 1],
            k=f[:, 2],
            color=color,
            opacity=opacity,
        )
        return [mesh_plotly]

    @staticmethod
    def pc(
        pc: np.ndarray,  # (n, 3)
        value: Optional[np.ndarray] = None,  # (n, )
        size: int = 1,
        color: Union[str, np.ndarray] = "red",  # (n, 3)
        color_map: str = "Viridis",
    ) -> list:
        """
        Visualize the point cloud with given color

        Parameters
        ----------
        pc: np.ndarray
            The point cloud with shape (n, 3)
        value: Optional[np.ndarray]
            The value of each point with shape (n, ), if None, use the color parameter
        size: int
            The size of the points
        color: Union[str, np.ndarray]
            The color of the points, if value is None, use this parameter
        color_map: str
            The color map to use, only used when value is not None

        Returns
        -------
        A list of plotly objects that can be shown in Vis.show
        """
        if value is None:
            if not isinstance(color, str):
                color = [
                    f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})"
                    for c in color
                ]
            marker = dict(size=size, color=color)
        else:
            marker = dict(size=size, color=value, colorscale=color_map, showscale=True)
        pc_plotly = go.Scatter3d(
            x=pc[:, 0],
            y=pc[:, 1],
            z=pc[:, 2],
            mode="markers",
            marker=marker,
        )
        return [pc_plotly]

    @staticmethod
    def show(
        plotly_list: list,
        path: Optional[str] = None,
    ) -> None:
        """
        Show the plotly objects or save them to a html file

        Parameters
        ----------
        plotly_list: list
            A list of plotly objects
        path: Optional[str]
            The path to save the html file, if None, show in the browser
        """
        fig = go.Figure(
            data=plotly_list, layout=go.Layout(scene=dict(aspectmode="data"))
        )
        if path is None:
            fig.show()
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fig.write_html(path)
            print(f"saved in {path}")
