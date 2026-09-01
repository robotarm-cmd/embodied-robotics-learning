import os
import time

import numpy as np

import pybullet as p
import pybullet_data

import torch
import torch.nn as nn

from robot_descriptions.loaders.pybullet import load_robot_description

MODEL_FILE = "cube_heatmap_model.pt"

SIM_DT = 1.0 / 240.0


WIDTH = 128
HEIGHT = 128

FOV_Y_DEG = 60.0

NEAR = 0.1
FAR = 3.0


# 2. position of camera
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
) # 拍出来图片在世界坐标系的位置，图片上方朝Y轴正向


TABLE_TOP_Z = 0.60

CUBE_HALF = 0.03


CUBE_POSITION = np.array(
    [
        0.55,
        0.14,
        TABLE_TOP_Z + CUBE_HALF
    ],
    dtype=float
)


TARGET_HEIGHT = 0.18

GAIN = 0.45

TOLERANCE = 0.038

MAX_STEPS = 20

MOVE_TIME = 0.60


HOME_Q_DEG = np.array(
    [
        0.0,
        -90.0,
        90.0,
        -90.0,
        -90.0,
        0.0
    ],
    dtype=float
)


class TinyHeatmapNet(
    nn.Module
):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.output_layer = nn.Conv2d(
            in_channels=16,
            out_channels=1,
            kernel_size=1
        )

        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()


    def forward(
        self,
        x
    ):
        x = self.conv1(
            x
        )

        x = self.relu(
            x
        )

        x = self.conv2(
            x
        )

        x = self.relu(
            x
        )

        x = self.conv3(
            x
        )

        x = self.relu(
            x
        )

        x = self.output_layer(
            x
        )

        x = self.sigmoid(
            x
        )

        return x

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
        raise ValueError("没有找到ur5机械臂")
        
    return joint_indices, joint_names

def find_ee_link(robot_id):
    candidate_names = ["tool0", "ee_link", "wrist_3_link"]
    number_of_joints = p.getNumJoints(robot_id)

    for candidate_name in candidate_names:
        for joint_index in range(number_of_joints):
            info = p.getJointInfo(robot_id, joint_index)

            link_name = info[12].decode("utf-8")

            if link_name == candidate_name:
                return joint_index, link_name

    raise RuntimeError("没有找到末端link")

def build_world():
    p.connect(
        p.GUI
    )

    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )

    p.setGravity(
        0.0,
        0.0,
        -9.81
    )

    p.setTimeStep(
        SIM_DT
    )

    p.loadURDF(
        "plane.urdf"
    )

    table_half = [
        0.40,
        0.32,
        TABLE_TOP_Z / 2.0
    ]

    table_center = [
        0.55,
        0.0,
        TABLE_TOP_Z / 2.0
    ]

    table_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=table_half
    )

    table_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=table_half,
        rgbaColor=[
            0.72,
            0.62,
            0.50,
            1.0
        ]
    )

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

    robot_id = load_robot_description(
        "ur5_official_description",
        useFixedBase=True
    )

    p.resetBasePositionAndOrientation(
        robot_id,
        [0.0, 0.0, TABLE_TOP_Z],
        [0.0, 0.0, 0.0, 1.0]
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

    return robot_id, cube_id

def reset_home(robot_id, joint_indices):
    q_home = np.deg2rad(HOME_Q_DEG)

    for joint_index, joint_angle in zip(joint_indices, q_home):
        p.resetJointState(
            robot_id,
            joint_index,
            float(joint_angle)
        )
    
    for _ in range(20):
        p.stepSimulation() # 让物理世界更新几次，让机器人、碰撞、视觉、物体状态等稳定一下。

def get_ee_pose(robot_id, ee_link):
    state = p.getLinkState(robot_id, ee_link, computeForwardKinematics=True)

    position = np.array(state[4], dtype=float)
    orientation = np.array(state[5], dtype=float)

    return position, orientation

def capture_rgbd():
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=CAMERA_EYE.tolist(),
        cameraTargetPosition=CAMERA_TARGET.tolist(),
        cameraUpVector=CAMERA_UP.tolist()
    ) # camera 在哪、朝哪

    projection_matrix = p.computeProjectionMatrixFOV(
        fov=FOV_Y_DEG,
        aspect=WIDTH / HEIGHT,
        nearVal=NEAR,
        farVal=FAR
    ) # camera 镜头投影

    image = p.getCameraImage(
        width=WIDTH,
        height=HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_TINY_RENDERER
    )

    rgba = np.asarray(
        image[2],
        dtype=np.uint8
    )

    rgba = rgba.reshape(
        HEIGHT,
        WIDTH,
        4
    )

    rgb = rgba[:, :, :3]

    depth_buffer = np.asarray(
        image[3],
        dtype=np.float32
    )

    depth_buffer = depth_buffer.reshape(HEIGHT, WIDTH)

    return rgb, depth_buffer

def prepare_cnn_input(rgb):
    image = rgb.astype(np.float32)

    image = image / 255.0

    image = np.transpose(image, (2, 0, 1))

    x = torch.from_numpy(image)
    x = x.unsqueeze(0)

    return x

def predict_pixel(model, rgb):
    x = prepare_cnn_input(rgb)

    with torch.no_grad():
        prediction = model(x)
    
    heatmap = prediction[0, 0].cpu().numpy()

    flat_index = np.argmax(heatmap)
    v, u = np.unravel_index(flat_index, heatmap.shape)
    score = float(heatmap[v, u])

    return int(u), int(v),  score

def depth_value_to_meters(depth_value):
    numerator = FAR * NEAR

    denominator = FAR - (FAR - NEAR) * depth_value

    return numerator / denominator

def get_camera_intrinsics():
    fov_y_rad = np.deg2rad(FOV_Y_DEG)
    fy = HEIGHT / (2.0 * np.tan(fov_y_rad / 2.0))

    fx = fy

    cx = (WIDTH - 1) / 2.0
    cy = (HEIGHT - 1) / 2.0

    return fx, fy, cx, cy

def pixel_to_world(u, v, depth_buffer):
    depth_value = float(depth_buffer[v, u])
    Zc = depth_value_to_meters(depth_value)
    fx, fy, cx, cy = get_camera_intrinsics()
    Xc = (u - cx) * Zc / fx
    Yc = (v - cy) * Zc / fy

    p_camera = np.array([Xc, Yc, Zc], dtype=np.float32)
    R_wc = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    p_surface_world = R_wc @ p_camera + CAMERA_EYE
    p_cube_world = p_surface_world.copy()

    p_cube_world[2] = p_cube_world[2] - CUBE_HALF

    return p_cube_world, p_camera, p_surface_world, Zc

def estimation_cube_position(model):
    rgb, depth_buffer = capture_rgbd()
    u, v, score = predict_pixel(model, rgb)
    p_cube_world, p_camera, p_surface_world, Zc = pixel_to_world(u, v, depth_buffer)
    print("CNN pixel:", (u, v))
    print("Heatmap max score:", round(score, 4))
    print("Depth Zc:", round(Zc, 4))
    print("Camera XYZ", np.round(p_camera, 4))
    print("Surface World XYZ", np.round(p_surface_world, 4))
    print("Cube Center World XYZ", np.round(p_cube_world, 4))

    return p_cube_world

def smooth_progress(t, total_time):
    tau = t / total_time

    tau = np.clip(tau, 0.0, 1.0)
    s = 10.0 * tau ** 3 -15.0 * tau ** 4 + 6.0 * tau ** 5

    return s

def get_joint_position(robot_id, joint_indices):
    states = p.getJointStates(robot_id, joint_indices)
    position = []

    for state in states:
        position.append(state[0])
    
    return np.array(position, dtype=float)

def move_joint_smooth(robot_id, joint_indices, q_goal, move_time):
    q_start = get_joint_position(robot_id, joint_indices)
    number_of_steps = int(move_time / SIM_DT)
    number_of_steps = max(number_of_steps, 1)
    
    for step_index in range(number_of_steps + 1):
        t = step_index * SIM_DT
        t = min(t, move_time)

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

def move_ee_to(robot_id, joint_indices, ee_link, target_position, target_orientation, move_time):
    ik_solution = p.calculateInverseKinematics(
        robot_id,
        ee_link,
        targetPosition=target_position.tolist(),
        targetOrientation=target_orientation.tolist(),
        maxNumIterations=200,
        residualThreshold=1e-5
    )

    q_goal = np.array(ik_solution[:len(joint_indices)], dtype=float)

    move_joint_smooth(robot_id, joint_indices, q_goal, move_time)

    actual_position, _ = get_ee_pose(robot_id, ee_link)
    final_error = np.linalg.norm(actual_position - target_position)

    return final_error

def cnn_visual_reach(model, robot_id, joint_indices, ee_link, hold_orientation):
    for step_index in range(MAX_STEPS):
        print("Visual step", step_index + 1)

        cube_position = estimation_cube_position(model)

        target_offset = np.array([0.0, 0.0, TARGET_HEIGHT], dtype=float)
        target_position = cube_position + target_offset
        ee_position, _ = get_ee_pose(robot_id, ee_link)

        error_vector = target_position - ee_position
        error = np.linalg.norm(error_vector)

        print("Target World XYZ:", np.round(target_position, 4))
        print("EE World XYZ:", np.round(ee_position, 4))
        print("Position error:", round(float(error), 4), "m")

        if error < TOLERANCE:
            print("\nSuccess!")

            return True
        
        next_position = ee_position + GAIN * error_vector
        print("Next World XYZ:", np.round(next_position, 4))

        ik_error = move_ee_to(
            robot_id,
            joint_indices,
            ee_link,
            next_position,
            hold_orientation,
            MOVE_TIME
        )
        
        print("IK execution error:", round(ik_error, 4))

    return False

def main():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError("没有找到模型文件:" + MODEL_FILE)

    model = TinyHeatmapNet()

    weights = torch.load(MODEL_FILE, map_location="cpu")
    model.load_state_dict(weights)

    model.eval()

    robot_id, cube_id = build_world()
    joint_indices, joint_names = get_arm_joints(robot_id)
    ee_link, ee_name = find_ee_link(robot_id)

    print("UR5 joints:", joint_names)

    print("End effectors:", ee_name)

    print("Cube id:", cube_id)

    reset_home(robot_id, joint_indices)
    _, hold_orientation = get_ee_pose(robot_id, ee_link)
    success = cnn_visual_reach(model, robot_id, joint_indices, ee_link, hold_orientation)
    print("\nresults:", success)

    input("按enter关闭pybullet")
    
    p.disconnect()

if __name__ == "__main__":
    main()