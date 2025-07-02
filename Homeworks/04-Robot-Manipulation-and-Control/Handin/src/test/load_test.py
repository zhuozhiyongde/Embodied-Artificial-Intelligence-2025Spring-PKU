from dataclasses import dataclass
import numpy as np
from src.utils import to_pose
from transforms3d.quaternions import quat2mat

@dataclass
class TestData:
    table_trans: np.ndarray
    table_size: np.ndarray
    obj_trans: np.ndarray
    obj_quat: np.ndarray
    quad_reset_pos: np.ndarray

EXAMPLE_TEST_DATA = [
    TestData(
        table_trans = np.array([0.6, 0.38, 0.72]),
        table_size = np.array([0.68, 0.36, 0.02]),
        obj_trans = np.array([0.5, 0.3, 0.82]),
        obj_quat = np.array([1.0, 0.0, 0.0, 0.0]),
        quad_reset_pos = np.array([2.0, -0.2, 0.278])
    ),

    TestData(
        table_trans = np.array([0.6, 0.45, 0.72]),
        table_size = np.array([0.72, 0.42, 0.02]),
        obj_trans = np.array([0.53, 0.35, 0.82]),
        obj_quat = np.array([0.966, 0.0, 0.0, 0.259]),
        quad_reset_pos = np.array([1.8, -0.25, 0.278])
    ),

    TestData(
        table_trans = np.array([0.55, 0.45, 0.68]),
        table_size = np.array([0.68, 0.36, 0.02]),
        obj_trans = np.array([0.5, 0.4, 0.82]),
        obj_quat = np.array([0.924, 0.0, 0.0, -0.383]),
        quad_reset_pos = np.array([1.9, -0.15, 0.278])
    ),

    TestData(
        table_trans = np.array([0.6, 0.47, 0.74]),
        table_size = np.array([0.68, 0.36, 0.02]),
        obj_trans = np.array([0.48, 0.43, 0.82]),
        obj_quat = np.array([0.174, 0.0, 0.0, 0.985]),
        quad_reset_pos = np.array([1.7, -0.1, 0.278])
    ),
]

def load_test_data(id = 0):
    testdata = EXAMPLE_TEST_DATA[id]
    
    return dict(
        table_pose = to_pose(trans=testdata.table_trans),
        table_size = testdata.table_size,
        obj_pose = to_pose(testdata.obj_trans, quat2mat(testdata.obj_quat)),
        quad_reset_pos = testdata.quad_reset_pos
    )