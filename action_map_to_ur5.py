import os
import time

import numpy as np

import pybullet as p
import pybullet_data

from robot_descriptions.loaders.pybullet import load_robot_description

import torch.nn as nn
import torch

SIM_DT = 1.0 / 240.0

WIDTH = 128
HEIGHT = 128

FOV_Y = 60.0

NEAR = 0.1
FAR = 3.0

CAMERA_EYE = np.array(
    [0.55, 0.0, 1.45],
    dtype=float
)

CAMERA_TARGET = np.array(
    [0.55, 0.0, 0.60],
    dtype=float
)

CAMERA_UP = np.array(
    [0.0, 1.0, 0.0],
    dtype=float
)

TABLE_TOP_Z = 0.60

CUBE_HALF = 0.03

PLACE_HALF = 0.05

PLACE_HALF_HEIGHT = 0.002

CUBE_POSITION = np.array(
    [
        0.55,
        0.14,
        TABLE_TOP_Z + CUBE_HALF
    ],
    dtype=float
)

PLACE_POSITION = np.array(
    [
        0.67,
        -0.12,
        TABLE_TOP_Z + PLACE_HALF_HEIGHT
    ],
    dtype=float
)

ROBOT_BASE_POSITION = np.array(
    [
        0.0,
        0.0,
        TABLE_TOP_Z
    ]
)

# Cube 顶面
PICK_SURFACE_Z = TABLE_TOP_Z + 2.0 * CUBE_HALF

# Place 顶面
PLACE_SURFACE_Z = TABLE_TOP_Z + 2.0 * PLACE_HALF_HEIGHT

OCCLUSION_PROBABILITY = 0.50

MIN_OCCLUSION_RATIO = 0.20
MAX_OCCLUSION_RATIO = 0.70

MAX_OCCLUSION_TRIES = 30

PICK_OCCLUSION_PROBILITY = 0.5

HOME_Q_DEG = np.array([0.0, -90.0, 90.0, -90.0, -90.0, 0.0], dtype=float)

HOVER_HEIGHT = 0.15
GRASP_CLEARANCE = 0.01
LIFT_HEIGHT = 0.12
MOVE_TIME = 1.0
ACTION_SCORE_THRESHOLD = 0.3
MAX_IK_ERROR = 0.03
SUCCESS_XY_TOLERANCE = 0.035

MODEL_FILE = "pick_place_unet.pt"

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        return x

class SmallUnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(3, 16)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.enc2 = ConvBlock(16, 32)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.bottleneck = ConvBlock(32, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(32, 16)
        self.output_layer = nn.Conv2d(16, 2, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        b = self.bottleneck(p2)
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat((d2, e2), dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat((d1, e1), dim=1))
        output = self.output_layer(d1)

        return output

def reset_home(robot_id, joint_indices):
    q_home = np.deg2rad(HOME_Q_DEG)

    for joint_index, joint_angle in zip(joint_indices, q_home):
        p.resetJointState(robot_id, joint_index, float(joint_angle))

    for _ in range(2):
        p.stepSimulation()

def get_arm_joints(robot_id):
    joint_indices = []
    joint_names = []

    number_of_joints = p.getNumJoints(robot_id)
    for joint_index in range(number_of_joints):
        info = p.getJointInfo(robot_id, joint_index)
        if info[2] == p.JOINT_REVOLUTE:
            joint_indices.append(joint_index)
            joint_names.append(info[1].decode("utf-8"))

    if len(joint_indices) < 6:
        raise ValueError("没有找到UR5的6个旋转关节")
    
    return joint_indices[:6], joint_names[:6]

def find_ee_link(robot_id):
    candidate_names = ["tool0", "ee_link", "wrist_3_link"]
    number_of_joints = p.getNumJoints(robot_id)
    for joint_index in range(number_of_joints):
        info = p.getJointInfo(robot_id, joint_index)
        name = info[12].decode("utf-8")
        if name in candidate_names:
            return joint_index
        
    raise ValueError("没有找到UR5末端关节")

def build_world():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0.0, 0.0, -9.81)
    p.setTimeStep(SIM_DT)
    p.loadURDF("plane.urdf")

    table_half = [0.40, 0.32, TABLE_TOP_Z / 2.0]
    table_center = [0.55, 0.0, TABLE_TOP_Z / 2.0]
    table_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=table_half)
    table_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=table_half, 
        rgbaColor=[
            0.72,
            0.62,
            0.50,
            1.0
        ])
    p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=table_collision,
        baseVisualShapeIndex=table_visual,
        basePosition=table_center
    )
    cube_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[
            CUBE_HALF,
            CUBE_HALF,
            CUBE_HALF
        ]
    )

    cube_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[
            CUBE_HALF,
            CUBE_HALF,
            CUBE_HALF
        ],
        rgbaColor=[
            0.05,
            0.20,
            1.0,
            1.0
        ]
    )

    cube_id = p.createMultiBody(
        baseMass=0.08,
        baseCollisionShapeIndex=cube_collision,
        baseVisualShapeIndex=cube_visual,
        basePosition=CUBE_POSITION.tolist()
    )

    place_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[
            PLACE_HALF,
            PLACE_HALF,
            PLACE_HALF_HEIGHT
        ],
        rgbaColor=[
            0.05,
            1.0,
            0.10,
            1.0
        ]
    )

    place_id = p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=place_visual,
        basePosition=PLACE_POSITION.tolist()
    )

    robot_id = load_robot_description(
        "ur5_official_description",
        useFixedBase=True
    )

    p.resetBasePositionAndOrientation(
        robot_id,
        ROBOT_BASE_POSITION.tolist(),
        [
            0.0,
            0.0,
            0.0,
            1.0
        ]
    )

    p.resetDebugVisualizerCamera(
        cameraDistance=1.55,
        cameraYaw=45,
        cameraPitch=-28,
        cameraTargetPosition=[
            0.50,
            0.0,
            0.60
        ]
    )

    return robot_id, cube_id, place_id

def get_ee_pose(robot_id, ee_link):
    state = p.getLinkState(robot_id, ee_link, computeForwardKinematics=True)
    position = np.array(state[4], dtype=float)
    orientation = np.array(state[5], dtype=float)

    return position, orientation

def capture_rgb():
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=CAMERA_EYE.tolist(),
        cameraTargetPosition=CAMERA_TARGET.tolist(),
        cameraUpVector=CAMERA_UP.tolist()
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=FOV_Y,
        aspect=WIDTH / HEIGHT,
        nearVal=NEAR,
        farVal=FAR
    )
    image = p.getCameraImage(
        width=WIDTH,
        height=HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_TINY_RENDERER
    )
    rgba = np.array(image[2], dtype=np.uint8)
    rgba = rgba.reshape(HEIGHT, WIDTH, 4)
    rgb = rgba[:,:,:3]

    return rgb

def prepare_network_input(rgb):
    image = rgb.astype(np.float32)
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    x = torch.from_numpy(image)
    x = x.unsqueeze(0)

    return x

def predict_action_pixels(model, rgb):
    x = prepare_network_input(rgb)
    with torch.no_grad():
        prediction = model(x)

    pick_heatmap = prediction[0, 0].cpu().numpy()
    place_heatmap = prediction[0, 1].cpu().numpy()

    pick_index = np.argmax(pick_heatmap)
    place_index = np.argmax(place_heatmap)

    v_pick, u_pick = np.unravel_index(pick_index, pick_heatmap.shape)
    v_place, u_place = np.unravel_index(place_index, place_heatmap.shape)

    pick_score = float(pick_heatmap[v_pick, u_pick])
    place_score = float(place_heatmap[v_place, u_place])

    return int(u_pick), int(v_pick), pick_score, int(u_place), int(v_place), place_score

def get_camera_intrinsic():
    fy = HEIGHT / (2.0 * np.tan(np.deg2rad(FOV_Y / 2.0)))
    fx = fy
    cx = (WIDTH - 1) / 2.0
    cy = (HEIGHT - 1) / 2.0

    return fx, fy, cx, cy

def build_camera_to_world_rotation():
    zcw = (CAMERA_TARGET - CAMERA_EYE) / np.linalg.norm(CAMERA_TARGET - CAMERA_EYE)
    xcw = np.cross(zcw, CAMERA_UP)
    xcw = xcw / np.linalg.norm(xcw)
    ycw = np.cross(zcw, xcw)
    R_wc = np.column_stack([xcw, ycw, zcw])
    
    return R_wc

def pixel_to_world_on_horizontal_plane(u, v, plane_z):
    fx, fy, cx, cy = get_camera_intrinsic()
    ray_camera = np.array([(u-cx) / fx, (v-cy) / fy, 1.0], dtype=float)
    R_wc = build_camera_to_world_rotation()

    ray_world = R_wc @ ray_camera
    if abs(ray_world[2]) < 1e-12:
        raise ValueError("Camera Ray 与目标水平面平行")

    scale = (plane_z - CAMERA_EYE[2]) / ray_world[2]
    if scale <= 0.0:
        raise ValueError("目标平面位于Camera Ray 后方")

    point_world = CAMERA_EYE + scale * ray_world

    return point_world

def estimate_action_world_points(model):
    rgb = capture_rgb()
    u_pick, v_pick, pick_score, u_place, v_place, place_score = predict_action_pixels(model, rgb)
    print("Pick Pixel:", (u_pick, v_pick), "Pick Score:", round(pick_score, 4))
    print("Place Pixel:", (u_place, v_place), "Place Score:", round(place_score, 4))

    if pick_score < ACTION_SCORE_THRESHOLD:
        print("Pick 置信度过低")
        return None
    if place_score < ACTION_SCORE_THRESHOLD:
        print("Place 置信度过低")
        return None

    pick_surface = pixel_to_world_on_horizontal_plane(u_pick, v_pick, PICK_SURFACE_Z)
    place_surface = pixel_to_world_on_horizontal_plane(u_place, v_place, PLACE_SURFACE_Z)

    return pick_surface, place_surface

def smooth_progress(t, total_time):
    tau = t / total_time

    tau = np.clip(tau, 0.0, 1.0)
    s = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5

    return s

def get_joint_positions(robot_id, joint_indices):
    states = p.getJointStates(robot_id, joint_indices)
    positions = []
    for state in states:
        positions.append(state[0])
    
    return np.array(positions, dtype=float)

def move_joints_smooth(robot_id, joint_indices, q_goal, move_time):
    q_start = get_joint_positions(robot_id, joint_indices)
    number_of_steps = max(int(move_time / SIM_DT), 1)
    for step_index in range(number_of_steps + 1):
        t = min(step_index * SIM_DT, move_time)
        s = smooth_progress(t, move_time)
        q_command = q_start + s * (q_goal - q_start)
        p.setJointMotorControlArray(
            robot_id,
            joint_indices,
            p.POSITION_CONTROL,
            targetPositions=q_command.tolist(),
            forces=[150.0] * len(joint_indices)
        )

        p.stepSimulation()
        time.sleep(SIM_DT)

def move_ee_to(
    robot_id,
    joint_indices,
    ee_link,
    target_position,
    target_orientation,
    move_time
):
    ik_solution = p.calculateInverseKinematics(
        robot_id,
        ee_link,
        targetPosition=target_position.tolist(),
        targetOrientation=target_orientation.tolist(),
        maxNumIterations=200,
        residualThreshold=1e-5
    )
    q_goal = np.array(ik_solution[:len(joint_indices)], dtype=float)
    move_joints_smooth(robot_id, joint_indices, q_goal, move_time)
    actual_postion, _ = get_ee_pose(robot_id, ee_link)
    error = np.linalg.norm(actual_postion - target_position)

    return float(error)

def attach_cube(robot_id, ee_link, cube_id):
    ee_position, ee_orientation = get_ee_pose(robot_id, ee_link)
    cube_position, cube_orientation = p.getBasePositionAndOrientation(cube_id)
    inverse_ee_position, inverse_ee_orientation = p.invertTransform(
        ee_position.tolist(),
        ee_orientation.tolist()
    )
    cube_in_ee_position, cube_in_ee_orientation = p.multiplyTransforms(
        inverse_ee_position,
        inverse_ee_orientation,
        cube_position,
        cube_orientation
    )
    constraint_id = p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=ee_link,
        childBodyUniqueId=cube_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0.0, 0.0, 0.0],
        parentFramePosition=cube_in_ee_position,
        childFramePosition=[0.0, 0.0, 0.0],
        parentFrameOrientation=cube_in_ee_orientation,
        childFrameOrientation=[0.0, 0.0, 0.0, 1.0]
    )
    p.changeConstraint(constraint_id, maxForce=200.0)

    return (
        constraint_id, 
        np.array(cube_in_ee_position, dtype=float), 
        np.array(cube_in_ee_orientation, dtype=float), 
        np.array(cube_orientation, dtype=float)
    )

def compute_ee_pose_for_cube_target(
    desired_cube_position,
    desired_cube_orientation,
    cube_in_ee_position,
    cube_in_ee_orientation
):
    ee_to_cube_position, ee_to_cube_orientation = p.invertTransform(
        cube_in_ee_position.tolist(), cube_in_ee_orientation.tolist()
    )
    ee_position, ee_orientation = p.multiplyTransforms(
        desired_cube_position,
        desired_cube_orientation,
        ee_to_cube_position,
        ee_to_cube_orientation
    )

    return np.array(ee_position, dtype=float), np.array(ee_orientation, dtype=float)

def verify_placement(cube_id, place_surface):
    cube_position, _ = p.getBasePositionAndOrientation(cube_id)
    cube_position = np.array(cube_position, dtype=float)
    xy_error = np.linalg.norm(cube_position[:2] - place_surface[:2])
    print("Final Cube XYZ", np.round(cube_position, 4))
    print("Place XY error:", round(float(xy_error), 4))

    return xy_error < SUCCESS_XY_TOLERANCE

def execute_pick_place(
    model,
    robot_id,
    joint_indices,
    ee_link,
    cube_id,
    hold_orientation
):
    # state 1 observe
    action_points = estimate_action_world_points(model)
    if action_points is None:
        return False
    pick_surface, place_surface = action_points
    pick_contact = pick_surface + np.array([0.0, 0.0, GRASP_CLEARANCE], dtype=float)
    pick_hover = pick_contact + np.array([0.0, 0.0, HOVER_HEIGHT], dtype=float)
    print("Pick contact:", np.round(pick_contact, 4))
    print("Pick hover:", np.round(pick_hover, 4))

    # state 2 pick hover
    error = move_ee_to(
        robot_id,
        joint_indices,
        ee_link,
        pick_hover,
        hold_orientation,
        MOVE_TIME
    )

    print("IK error:", round(error, 4))

    if error > MAX_IK_ERROR:
        return False

    # state 3 descend pick
    error = move_ee_to(
        robot_id,
        joint_indices,
        ee_link,
        pick_contact,
        hold_orientation,
        MOVE_TIME
    )

    print("IK error:", round(error, 4))

    if error > MAX_IK_ERROR:
        return False

    # state 4 grasp
    (
        constraint_id,
        cube_in_ee_position,
        cube_in_ee_orientation,
        cube_orientation
    ) = attach_cube(robot_id, ee_link, cube_id)

    for _ in range(10):
        p.stepSimulation()

    # state 5 lift
    current_ee_position, current_ee_orientation = get_ee_pose(robot_id, ee_link)
    lift_postion = current_ee_position + np.array([0.0, 0.0, LIFT_HEIGHT], dtype=float)
    error = move_ee_to(
        robot_id,
        joint_indices,
        ee_link,
        lift_postion,
        current_ee_orientation,
        MOVE_TIME
    )

    print("IK error:", round(error, 4))

    if error > MAX_IK_ERROR:
        p.removeConstraint(constraint_id)
        return False

    desired_cube_position = np.array([
        place_surface[0],
        place_surface[1],
        place_surface[2] + CUBE_HALF
    ], dtype=float)

    place_ee_position, place_ee_orientation = compute_ee_pose_for_cube_target(
        desired_cube_position,
        cube_orientation,
        cube_in_ee_position,
        cube_in_ee_orientation
    )

    place_hover_position = place_ee_position + np.array([0.0, 0.0, HOVER_HEIGHT], dtype=float)
    print("Desired Cube XYZ:", np.round(desired_cube_position, 4))
    print("Place EE XYZ:", np.round(place_ee_position, 4))

    # state 6 place hover
    error = move_ee_to(
        robot_id,
        joint_indices,
        ee_link,
        place_hover_position,
        place_ee_orientation,
        MOVE_TIME
    )

    print("IK error:", round(error, 4))

    if error > MAX_IK_ERROR:
        p.removeConstraint(constraint_id)
        return False

    # state 7 descend place
    error = move_ee_to(
        robot_id,
        joint_indices,
        ee_link,
        place_ee_position,
        place_ee_orientation,
        MOVE_TIME
    )

    print("IK error:", round(error, 4))

    if error > MAX_IK_ERROR:
        p.removeConstraint(constraint_id)
        return False

    # state 8 release
    p.removeConstraint(constraint_id)
    for _ in range(120):
        p.stepSimulation()
        time.sleep(SIM_DT)

    # state 9 retreat
    current_ee_position, current_ee_orientation = get_ee_pose(robot_id, ee_link)
    retreat_position = current_ee_position + np.array([0.0, 0.0, HOVER_HEIGHT], dtype=float)

    move_ee_to(
        robot_id,
        joint_indices,
        ee_link,
        retreat_position,
        current_ee_orientation,
        MOVE_TIME
    )

    # state 10 verify
    success = verify_placement(
        cube_id,
        place_surface
    )

    return success

def main():
    if not os.path.exists(
        MODEL_FILE
    ):
        raise FileNotFoundError("没有找到模型文件")

    model = SmallUnet()

    weights = torch.load(MODEL_FILE, map_location="cpu")
    model.load_state_dict(weights)
    model.eval()

    robot_id, cube_id, place_id = build_world()
    joint_indices, joint_names = get_arm_joints(robot_id)
    ee_link = find_ee_link(robot_id)

    print("UR5 joints:", joint_names)
    print("Cube id:", cube_id)
    print("Place id:", place_id)

    reset_home(robot_id, joint_indices)

    _, hold_orientation = get_ee_pose(robot_id, ee_link)

    success = execute_pick_place(
        model,
        robot_id,
        joint_indices,
        ee_link,
        cube_id,
        hold_orientation
    )

    if success:
        print("机械臂pick and place成功")

    p.disconnect()

if __name__ == "__main__":
    main()




