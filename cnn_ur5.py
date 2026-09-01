import os
import time

import numpy as np

import pybullet as p
import pybullet_data

import torch
import torch.nn as nn

from robot_descriptions.loaders.pybullet import (
    load_robot_description
)


# ============================================================
# 1. model
# ============================================================

MODEL_FILE = "cube_heatmap_model.pt"

SIM_DT = 1.0 / 240.0


# ============================================================
# 2. camera image
# ============================================================

WIDTH = 128
HEIGHT = 128

FOV_Y_DEG = 60.0

NEAR = 0.1
FAR = 3.0


# ============================================================
# 3. position of camera
# ============================================================

CAMERA_EYE = np.array(
    [
        0.55,
        0.0,
        1.45
    ],
    dtype=float
)


CAMERA_TARGET = np.array(
    [
        0.55,
        0.0,
        0.60
    ],
    dtype=float
)


CAMERA_UP = np.array(
    [
        0.0,
        1.0,
        0.0
    ],
    dtype=float
)
# 拍出来图片在世界坐标系的位置，
# 图片上方朝 Y 轴正向


# ============================================================
# 4. table and cube
# ============================================================

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


# ============================================================
# 5. visual reaching
# ============================================================

TARGET_HEIGHT = 0.18

GAIN = 0.45

TOLERANCE = 0.038

MAX_STEPS = 20

MOVE_TIME = 0.60


# ============================================================
# 6. UR5 home position
# ============================================================

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


# ============================================================
# 7. CNN
# ============================================================

class TinyHeatmapNet(
    nn.Module
):

    def __init__(
        self
    ):

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


# ============================================================
# 8. normalize vector
# ============================================================

def normalize_vector(
    vector
):

    vector = np.asarray(
        vector,
        dtype=np.float64
    )


    norm = np.linalg.norm(
        vector
    )


    if norm < 1e-12:

        raise ValueError(
            "向量长度太小，无法归一化。"
        )


    return (
        vector
        /
        norm
    )


# ============================================================
# 9. build R_cw
#
# world coordinate
# ->
# camera coordinate
# ============================================================

def build_world_to_camera_rotation():

    # --------------------------------------------------------
    # Camera z axis:
    #
    # CAMERA_EYE
    # ->
    # CAMERA_TARGET
    #
    # z_c is camera forward
    # --------------------------------------------------------

    z_camera_world = (
        CAMERA_TARGET
        -
        CAMERA_EYE
    )


    z_camera_world = (
        normalize_vector(
            z_camera_world
        )
    )


    # --------------------------------------------------------
    # Camera x axis:
    #
    # image right
    # --------------------------------------------------------

    x_camera_world = np.cross(
        z_camera_world,
        CAMERA_UP
    )


    x_camera_world = (
        normalize_vector(
            x_camera_world
        )
    )


    # --------------------------------------------------------
    # Camera y axis:
    #
    # image downward
    # --------------------------------------------------------

    y_camera_world = np.cross(
        z_camera_world,
        x_camera_world
    )


    y_camera_world = (
        normalize_vector(
            y_camera_world
        )
    )


    # --------------------------------------------------------
    # R_cw:
    #
    # world -> camera
    #
    #       [xc^T]
    # Rcw = [yc^T]
    #       [zc^T]
    # --------------------------------------------------------

    R_cw = np.vstack(
        [
            x_camera_world,
            y_camera_world,
            z_camera_world
        ]
    )


    return R_cw


# ============================================================
# 10. build camera intrinsic matrix K
# ============================================================

def build_camera_intrinsic_matrix():

    fov_y_rad = np.deg2rad(
        FOV_Y_DEG
    )


    fy = (
        HEIGHT
        /
        (
            2.0
            *
            np.tan(
                fov_y_rad
                /
                2.0
            )
        )
    )


    fx = fy


    # --------------------------------------------------------
    # 与遮挡训练代码保持相同
    # --------------------------------------------------------

    cx = (
        (WIDTH - 1) 
        /
        2.0
    )


    cy = (
        (HEIGHT - 1)
        /
        2.0
    )


    K = np.array(
        [
            [
                fx,
                0.0,
                cx
            ],

            [
                0.0,
                fy,
                cy
            ],

            [
                0.0,
                0.0,
                1.0
            ]
        ],
        dtype=np.float64
    )


    return K


# ============================================================
# 11. get UR5 joints
# ============================================================

def get_arm_joints(
    robot_id
):

    joint_indices = []

    joint_names = []


    number_of_joints = (
        p.getNumJoints(
            robot_id
        )
    )


    for joint_index in range(
        number_of_joints
    ):

        info = p.getJointInfo(
            robot_id,
            joint_index
        )


        if (
            info[2]
            ==
            p.JOINT_REVOLUTE
        ):

            joint_indices.append(
                joint_index
            )


            joint_names.append(
                info[1]
                .decode(
                    "utf-8"
                )
            )


    if len(
        joint_indices
    ) < 6:

        raise ValueError(
            "没有找到ur5机械臂"
        )


    return (
        joint_indices,
        joint_names
    )


# ============================================================
# 12. find end-effector link
# ============================================================

def find_ee_link(
    robot_id
):

    candidate_names = [
        "tool0",
        "ee_link",
        "wrist_3_link"
    ]


    number_of_joints = (
        p.getNumJoints(
            robot_id
        )
    )


    # --------------------------------------------------------
    # 先找 tool0
    # 找不到再找 ee_link
    # 最后找 wrist_3_link
    # --------------------------------------------------------

    for candidate_name in candidate_names:

        for joint_index in range(
            number_of_joints
        ):

            info = p.getJointInfo(
                robot_id,
                joint_index
            )


            link_name = (
                info[12]
                .decode(
                    "utf-8"
                )
            )


            if (
                link_name
                ==
                candidate_name
            ):

                return (
                    joint_index,
                    link_name
                )


    raise RuntimeError(
        "没有找到末端link"
    )


# ============================================================
# 13. build PyBullet world
# ============================================================

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


    # ========================================================
    # table
    #
    # 保持你的原始数据不变
    # ========================================================

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


    table_collision = (
        p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=
            table_half
        )
    )


    table_visual = (
        p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=
            table_half,
            rgbaColor=[
                0.72,
                0.62,
                0.50,
                1.0
            ]
        )
    )


    p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=
        table_collision,
        baseVisualShapeIndex=
        table_visual,
        basePosition=
        table_center
    )


    # ========================================================
    # cube
    #
    # 保持你的原始数据不变
    # ========================================================

    cube_collision = (
        p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[
                CUBE_HALF,
                CUBE_HALF,
                CUBE_HALF
            ]
        )
    )


    cube_visual = (
        p.createVisualShape(
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
    )


    cube_id = (
        p.createMultiBody(
            baseMass=0.08,
            baseCollisionShapeIndex=
            cube_collision,
            baseVisualShapeIndex=
            cube_visual,
            basePosition=
            CUBE_POSITION.tolist()
        )
    )


    # ========================================================
    # UR5
    # ========================================================

    robot_id = (
        load_robot_description(
            "ur5_official_description",
            useFixedBase=True
        )
    )


    p.resetBasePositionAndOrientation(
        robot_id,
        [
            0.0,
            0.0,
            TABLE_TOP_Z
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0
        ]
    )


    # ========================================================
    # GUI camera
    # ========================================================

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


    return (
        robot_id,
        cube_id
    )


# ============================================================
# 14. reset home
# ============================================================

def reset_home(
    robot_id,
    joint_indices
):

    q_home = np.deg2rad(
        HOME_Q_DEG
    )


    for (
        joint_index,
        joint_angle
    ) in zip(
        joint_indices,
        q_home
    ):

        p.resetJointState(
            robot_id,
            joint_index,
            float(
                joint_angle
            )
        )


    for _ in range(
        20
    ):

        p.stepSimulation()


# ============================================================
# 15. get end-effector pose
# ============================================================

def get_ee_pose(
    robot_id,
    ee_link
):

    state = p.getLinkState(
        robot_id,
        ee_link,
        computeForwardKinematics=True
    )


    position = np.array(
        state[4],
        dtype=float
    )


    orientation = np.array(
        state[5],
        dtype=float
    )


    return (
        position,
        orientation
    )


# ============================================================
# 16. capture RGB + depth
#
# 保持原始函数结构
# depth 仍然保留，但新的定位方法不再依赖它
# ============================================================

def capture_rgbd():

    view_matrix = (
        p.computeViewMatrix(
            cameraEyePosition=
            CAMERA_EYE.tolist(),

            cameraTargetPosition=
            CAMERA_TARGET.tolist(),

            cameraUpVector=
            CAMERA_UP.tolist()
        )
    )


    projection_matrix = (
        p.computeProjectionMatrixFOV(
            fov=
            FOV_Y_DEG,

            aspect=
            WIDTH
            /
            HEIGHT,

            nearVal=
            NEAR,

            farVal=
            FAR
        )
    )


    image = (
        p.getCameraImage(
            width=
            WIDTH,

            height=
            HEIGHT,

            viewMatrix=
            view_matrix,

            projectionMatrix=
            projection_matrix,

            renderer=
            p.ER_BULLET_HARDWARE_OPENGL
        )
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


    rgb = (
        rgba[
            :,
            :,
            :3
        ]
    )


    depth_buffer = np.asarray(
        image[3],
        dtype=np.float32
    )


    depth_buffer = (
        depth_buffer.reshape(
            HEIGHT,
            WIDTH
        )
    )


    return (
        rgb,
        depth_buffer
    )


# ============================================================
# 17. prepare CNN input
# ============================================================

def prepare_cnn_input(
    rgb
):

    image = rgb.astype(
        np.float32
    )


    image = (
        image
        /
        255.0
    )


    image = np.transpose(
        image,
        (
            2,
            0,
            1
        )
    )


    x = torch.from_numpy(
        image
    )


    x = x.unsqueeze(
        0
    )


    return x


# ============================================================
# 18. CNN predict pixel
# ============================================================

def predict_pixel(
    model,
    rgb
):

    x = prepare_cnn_input(
        rgb
    )


    with torch.no_grad():

        prediction = model(
            x
        )


    heatmap = (
        prediction[
            0,
            0
        ]
        .cpu()
        .numpy()
    )


    flat_index = np.argmax(
        heatmap
    )


    (
        v,
        u
    ) = np.unravel_index(
        flat_index,
        heatmap.shape
    )


    score = float(
        heatmap[
            v,
            u
        ]
    )


    return (
        int(u),
        int(v),
        score
    )


# ============================================================
# 19. pixel -> cube center world position
#
# 不再使用 depth
#
# pixel
# ->
# camera ray
# ->
# world ray
# ->
# 与 z = cube center z 平面求交
# ============================================================

def pixel_to_world(
    u,
    v,
    R_cw,
    K
):

    # ========================================================
    # 1. pixel homogeneous coordinate
    #
    # [u]
    # [v]
    # [1]
    # ========================================================

    pixel_homogeneous = np.array(
        [
            float(u),
            float(v),
            1.0
        ],
        dtype=np.float64
    )


    # ========================================================
    # 2. pixel -> camera ray
    #
    # d_c = K^-1 p
    # ========================================================

    K_inverse = np.linalg.inv(
        K
    )


    ray_camera = (
        K_inverse
        @
        pixel_homogeneous
    )


    # ========================================================
    # 3. R_cw:
    #
    # world -> camera
    #
    # therefore
    #
    # R_wc = R_cw^T
    #
    # camera -> world
    # ========================================================

    R_wc = (
        R_cw.T
    )


    # ========================================================
    # 4. camera ray -> world ray
    # ========================================================

    ray_world = (
        R_wc
        @
        ray_camera
    )


    # ========================================================
    # 5. cube center height
    #
    # z = 0.60 + 0.03
    #
    # 保持原始数据
    # ========================================================

    cube_center_z = (
        TABLE_TOP_Z
        +
        CUBE_HALF
    )


    # ========================================================
    # 6. world ray:
    #
    # P(t) = CAMERA_EYE + t * ray_world
    #
    # We require:
    #
    # Pz(t) = cube_center_z
    # ========================================================

    if abs(
        ray_world[2]
    ) < 1e-12:

        raise ValueError(
            "相机射线与方块中心平面平行。"
        )


    t = (
        cube_center_z
        -
        CAMERA_EYE[2]
    ) / ray_world[2]


    # ========================================================
    # 如果 t <= 0
    # 交点位于相机后面
    # ========================================================

    if t <= 0.0:

        raise ValueError(
            "像素射线与目标平面的交点位于相机后方。"
        )


    # ========================================================
    # 7. cube center world position
    # ========================================================

    p_cube_world = (
        CAMERA_EYE
        +
        t
        *
        ray_world
    )


    return (
        p_cube_world,
        ray_camera,
        ray_world,
        t
    )


# ============================================================
# 20. estimate cube position
# ============================================================

def estimation_cube_position(
    model,
    R_cw,
    K
):

    # --------------------------------------------------------
    # depth_buffer 仍然从原函数返回
    # 但这里不再使用 depth 估计 cube position
    # --------------------------------------------------------

    (
        rgb,
        depth_buffer
    ) = (
        capture_rgbd()
    )


    (
        u,
        v,
        score
    ) = (
        predict_pixel(
            model,
            rgb
        )
    )


    (
        p_cube_world,
        ray_camera,
        ray_world,
        ray_scale
    ) = (
        pixel_to_world(
            u,
            v,
            R_cw,
            K
        )
    )


    print(
        "CNN pixel:",
        (
            u,
            v
        )
    )


    print(
        "Heatmap max score:",
        round(
            score,
            4
        )
    )


    print(
        "Camera Ray:",
        np.round(
            ray_camera,
            4
        )
    )


    print(
        "World Ray:",
        np.round(
            ray_world,
            4
        )
    )


    print(
        "Ray scale:",
        round(
            float(
                ray_scale
            ),
            4
        )
    )


    print(
        "Cube Center World XYZ:",
        np.round(
            p_cube_world,
            4
        )
    )


    return p_cube_world


# ============================================================
# 21. smooth progress
# ============================================================

def smooth_progress(
    t,
    total_time
):

    tau = (
        t
        /
        total_time
    )


    tau = np.clip(
        tau,
        0.0,
        1.0
    )


    s = (
        10.0
        *
        tau ** 3

        -

        15.0
        *
        tau ** 4

        +

        6.0
        *
        tau ** 5
    )


    return s


# ============================================================
# 22. get joint positions
# ============================================================

def get_joint_position(
    robot_id,
    joint_indices
):

    states = p.getJointStates(
        robot_id,
        joint_indices
    )


    position = []


    for state in states:

        position.append(
            state[0]
        )


    return np.array(
        position,
        dtype=float
    )


# ============================================================
# 23. smooth joint motion
# ============================================================

def move_joint_smooth(
    robot_id,
    joint_indices,
    q_goal,
    move_time
):

    q_start = (
        get_joint_position(
            robot_id,
            joint_indices
        )
    )


    number_of_steps = int(
        move_time
        /
        SIM_DT
    )


    number_of_steps = max(
        number_of_steps,
        1
    )


    for step_index in range(
        number_of_steps
        +
        1
    ):

        t = (
            step_index
            *
            SIM_DT
        )


        t = min(
            t,
            move_time
        )


        s = smooth_progress(
            t,
            move_time
        )


        q_command = (
            q_start
            +
            s
            *
            (
                q_goal
                -
                q_start
            )
        )


        p.setJointMotorControlArray(
            robot_id,
            joint_indices,
            p.POSITION_CONTROL,
            targetPositions=
            q_command.tolist(),
            forces=[
                150.0
            ]
            *
            len(
                joint_indices
            )
        )


        p.stepSimulation()


        time.sleep(
            SIM_DT
        )


# ============================================================
# 24. move end effector to target
# ============================================================

def move_ee_to(
    robot_id,
    joint_indices,
    ee_link,
    target_position,
    target_orientation,
    move_time
):

    ik_solution = (
        p.calculateInverseKinematics(
            robot_id,
            ee_link,
            targetPosition=
            target_position.tolist(),
            targetOrientation=
            target_orientation.tolist(),
            maxNumIterations=200,
            residualThreshold=1e-5
        )
    )


    q_goal = np.array(
        ik_solution[
            :
            len(
                joint_indices
            )
        ],
        dtype=float
    )


    move_joint_smooth(
        robot_id,
        joint_indices,
        q_goal,
        move_time
    )


    actual_position, _ = (
        get_ee_pose(
            robot_id,
            ee_link
        )
    )


    final_error = np.linalg.norm(
        actual_position
        -
        target_position
    )


    return final_error


# ============================================================
# 25. CNN visual reaching
# ============================================================

def cnn_visual_reach(
    model,
    robot_id,
    joint_indices,
    ee_link,
    hold_orientation,
    R_cw,
    K
):

    for step_index in range(
        MAX_STEPS
    ):

        print(
            "\nVisual step",
            step_index + 1
        )


        # ====================================================
        # CNN:
        #
        # RGB
        # ->
        # heatmap
        # ->
        # pixel
        # ->
        # world ray
        # ->
        # cube center world position
        # ====================================================

        cube_position = (
            estimation_cube_position(
                model,
                R_cw,
                K
            )
        )


        # ====================================================
        # target is above cube
        # ====================================================

        target_offset = np.array(
            [
                0.0,
                0.0,
                TARGET_HEIGHT
            ],
            dtype=float
        )


        target_position = (
            cube_position
            +
            target_offset
        )


        # ====================================================
        # current end-effector position
        # ====================================================

        ee_position, _ = (
            get_ee_pose(
                robot_id,
                ee_link
            )
        )


        # ====================================================
        # position error
        # ====================================================

        error_vector = (
            target_position
            -
            ee_position
        )


        error = np.linalg.norm(
            error_vector
        )


        print(
            "Target World XYZ:",
            np.round(
                target_position,
                4
            )
        )


        print(
            "EE World XYZ:",
            np.round(
                ee_position,
                4
            )
        )


        print(
            "Position error:",
            round(
                float(
                    error
                ),
                4
            ),
            "m"
        )


        # ====================================================
        # reached
        # ====================================================

        if (
            error
            <
            TOLERANCE
        ):

            print(
                "\nSuccess!"
            )


            return True


        # ====================================================
        # move only part of error
        #
        # next =
        # current + gain * error
        # ====================================================

        next_position = (
            ee_position
            +
            GAIN
            *
            error_vector
        )


        print(
            "Next World XYZ:",
            np.round(
                next_position,
                4
            )
        )


        # ====================================================
        # IK movement
        # ====================================================

        ik_error = (
            move_ee_to(
                robot_id,
                joint_indices,
                ee_link,
                next_position,
                hold_orientation,
                MOVE_TIME
            )
        )


        print(
            "IK execution error:",
            round(
                ik_error,
                4
            )
        )


    return False


# ============================================================
# 26. main
# ============================================================

def main():

    # ========================================================
    # 1. model file
    # ========================================================

    if not os.path.exists(
        MODEL_FILE
    ):

        raise FileNotFoundError(
            "没有找到模型文件:"
            +
            MODEL_FILE
        )


    # ========================================================
    # 2. create CNN
    # ========================================================

    model = (
        TinyHeatmapNet()
    )


    # ========================================================
    # 3. load weights
    # ========================================================

    weights = torch.load(
        MODEL_FILE,
        map_location="cpu"
    )


    model.load_state_dict(
        weights
    )


    model.eval()


    # ========================================================
    # 4. build R_cw
    #
    # world -> camera
    # ========================================================

    R_cw = (
        build_world_to_camera_rotation()
    )


    # ========================================================
    # 5. build K
    # ========================================================

    K = (
        build_camera_intrinsic_matrix()
    )


    print(
        "\nR_cw ="
    )


    print(
        R_cw
    )


    print(
        "\nK ="
    )


    print(
        K
    )


    # ========================================================
    # 6. PyBullet world
    # ========================================================

    (
        robot_id,
        cube_id
    ) = (
        build_world()
    )


    # ========================================================
    # 7. UR5 joints
    # ========================================================

    (
        joint_indices,
        joint_names
    ) = (
        get_arm_joints(
            robot_id
        )
    )


    # ========================================================
    # 8. end effector
    # ========================================================

    (
        ee_link,
        ee_name
    ) = (
        find_ee_link(
            robot_id
        )
    )


    print(
        "UR5 joints:",
        joint_names
    )


    print(
        "End effectors:",
        ee_name
    )


    print(
        "Cube id:",
        cube_id
    )


    # ========================================================
    # 9. reset robot home
    # ========================================================

    reset_home(
        robot_id,
        joint_indices
    )


    # ========================================================
    # 10. keep current end-effector orientation
    # ========================================================

    (
        _,
        hold_orientation
    ) = (
        get_ee_pose(
            robot_id,
            ee_link
        )
    )


    # ========================================================
    # 11. CNN visual reach
    # ========================================================

    success = (
        cnn_visual_reach(
            model,
            robot_id,
            joint_indices,
            ee_link,
            hold_orientation,
            R_cw,
            K
        )
    )


    print(
        "\nresults:",
        success
    )


    input(
        "按enter关闭pybullet"
    )


    p.disconnect()


# ============================================================
# 27. program entry
# ============================================================

if __name__ == "__main__":

    main()
