FULL_COLLISIONS_FLAT_TERRAIN_XML = "src/quad_highlevel/assets/unitree_go2/xmls/scene_mjx_fullcollisions_with_aruco_flat_terrain.xml"
POLICY_ONNX_PATH = "src/quad_highlevel/model_weights/go2_loco_v9.onnx"

FEET_SITES = [
    "FR",
    "FL",
    "RR",
    "RL",
]

FEET_GEOMS = [
    "FR",
    "FL",
    "RR",
    "RL",
]

FEET_POS_SENSOR = [f"{site}_pos" for site in FEET_SITES]

ROOT_BODY = "base"

UPVECTOR_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"

# fmt: off
DEFAULT_BASE_POSE = [
    0.0, 0.0, 0.278,
    1.0, 0.0, 0.0, 0.0
]

DEFAULT_JOINT_ANGLES = [
    0.1, 0.9, -1.8,
    -0.1, 0.9, -1.8,
    0.1, 0.9, -1.8,
    -0.1, 0.9, -1.8,
]
