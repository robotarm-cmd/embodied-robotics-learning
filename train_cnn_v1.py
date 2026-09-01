from math import nan

import numpy as np
import matplotlib.pyplot as plt

from sympy.core.numbers import NaN
import torch
from torch.cpu import is_available
import torch.nn as nn

import pybullet as p
import pybullet_data

from robot_descriptions.loaders.pybullet import load_robot_description
from torch.types import Device

# 1. camera parameters
WIDTH = 128
HEIGHT = 128

FOV_Y = 60.0
FOV_X = FOV_Y

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

# 3. desk and cube parameters
TABLE_TOP_Z = 0.60
CUBE_HALF = 0.03

# 4. training parameters 
TRAIN_SAMPLES = 200
VAL_SAMPLES = 40

BATCH_SIZE = 16

EPOCHS = 20

LEARNING_RATE = 0.001

# new: 遮挡参数
OCCLUSION_PROBABILITY = 0.6
MIN_OCCLUSION_RATIO = 0.20
MAX_OCCLUSION_RATIO = 1.0
MAX_OCCLUSION_TRIES = 10

MODEL_FILE = "cube_heatmap_model.pt"

# 5. UR5 initial degree
HOME_Q_DEG = np.array(
    [0.0, -90.0, 90.0, -90.0, -90.0, 0.0],
    dtype=float
)

# new: 建立旋转矩阵
def build_world_to_camera_rotation(): # 世界在camera坐标系中的方向

    z_camera_world = CAMERA_TARGET - CAMERA_EYE
    z_camera_world = z_camera_world / np.linalg.norm(z_camera_world)
    x_camera_world = np.cross(z_camera_world, CAMERA_UP)
    x_camera_world = x_camera_world / np.linalg.norm(x_camera_world)
    y_camera_world = np.cross(z_camera_world, x_camera_world)
    
    return np.vstack((x_camera_world,y_camera_world,z_camera_world))

# new: 建立内参矩阵
def build_camera_intrinsic_matrix():
    fov_y_rad = np.deg2rad(FOV_Y)
    fy = HEIGHT / (2 * np.tan(fov_y_rad / 2))
    fx = fy
    cx = (WIDTH - 1) / 2.0
    cy = (HEIGHT - 1)/ 2.0

    return np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]])

# 6. find joints of ur5
def get_arm_joints(robot_id):
    joint_indices = []

    number_of_joints = p.getNumJoints(robot_id)

    for joint_index in range(number_of_joints):
        info = p.getJointInfo(
            robot_id,
            joint_index
        )

        if info[2] == p.JOINT_REVOLUTE:
            joint_indices.append(
                joint_index
            )
    
    if len(joint_indices) < 6:
        raise ValueError("没有找到UR5的6个旋转关节")
    
    return joint_indices[:6]

# 7. initiate UR5
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

# new: 找UR5末端索引
def find_end_effector_link(robot_id):
    candidate_names = ["tool0", "ee_link", "wrist_3_link"]
    number_of_joints = p.getNumJoints(robot_id)
    for joint_index in range(number_of_joints):
        joint_info = p.getJointInfo(robot_id, joint_index)
        link_name = joint_info[12].decode("utf-8")
        if link_name in candidate_names:
            return joint_index
    
    raise ValueError("没有找到UR5的末端关节")

# 8. build training world
# revise: 加入返回值robot_id, table_id, 并加入函数寻找UR5旋转轴索引, 末端索引, 末端方位
def build_training_world():
    p.connect(p.DIRECT) 

    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )

    p.setGravity(
        0.0,
        0.0,
        -9.81
    )

    p.setTimeStep(
        1.0 / 240.0
    )

    p.loadURDF(
        "plane.urdf"
    )

    table_half = [
        0.40,
        0.32,
        TABLE_TOP_Z / 2
    ]

    table_center = [
        0.55,
        0.0,
        TABLE_TOP_Z / 2
    ]

    table_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=table_half
    ) # 物理防碰撞

    table_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=table_half,
        rgbaColor=[
            0.72,
            0.62,
            0.50,
            1.0
        ]
    ) # 视觉外观

    table_id = p.createMultiBody(
        baseMass=0.0, # 固定桌子
        baseCollisionShapeIndex=table_collision,
        baseVisualShapeIndex=table_visual,
        basePosition=table_center
    ) # 真正创建一个物体，并把“质量、碰撞形状、视觉形状、位置”等信息组合到一起。

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
        baseMass=0.0,
        baseCollisionShapeIndex=cube_collision,
        baseVisualShapeIndex=cube_visual,
        basePosition=[
            0.55,
            0.0,
            TABLE_TOP_Z + CUBE_HALF
        ]
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

    joint_indices = get_arm_joints(robot_id)
    reset_home(
        robot_id,
        joint_indices
    )

    end_effector_link = find_end_effector_link(robot_id)
    ee_state = p.getLinkState(robot_id, end_effector_link, computeForwardKinematics=True)
    home_ee_orientation = ee_state[5]

    return table_id, robot_id, cube_id, joint_indices, end_effector_link, home_ee_orientation

# 9. make heatmap
def make_gaussian_heatmap(
    u_center,
    v_center,
    sigma=4.0
):
    u_coordinates = np.arange(WIDTH)
    v_coordinates = np.arange(HEIGHT)

    grid_u, grid_v = np.meshgrid(
        u_coordinates,
        v_coordinates
    )

    squared_distance = (grid_u - u_center)**2 + (grid_v - v_center)**2
    heat_map = np.exp(-squared_distance / (2 * sigma ** 2))

    return heat_map.astype(
        np.float32
    )

# 10. capture rgb and segmentation
def capture_rgb_and_segmentation():
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=CAMERA_EYE.tolist(),
        cameraTargetPosition=CAMERA_TARGET.tolist(),
        cameraUpVector=CAMERA_UP.tolist()
    ) # camera 在哪、朝哪

    projection_matrix = p.computeProjectionMatrixFOV(
        fov=FOV_Y,
        aspect=WIDTH / HEIGHT,
        nearVal=NEAR,
        farVal=FAR
    ) # camera 镜头投影

    image = p.getCameraImage(
        width=WIDTH,
        height=HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
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

    segmentation = np.asarray(
        image[4],
        dtype=np.int32
    )

    segmentation = segmentation.reshape(
        HEIGHT,
        WIDTH,
    )

    return rgb, segmentation

# new: u, v新的计算方式，就算被遮挡仍然知道u, v坐标
def world_to_pixel(cube_position, R_cw, K):
    position_rw = cube_position - CAMERA_EYE
    pc = R_cw @ position_rw
    Z_c = pc[2]
    if Z_c <= 0:
        return None

    pixel_homogeneous = K @ pc
    u = pixel_homogeneous[0] / pixel_homogeneous[2]
    v = pixel_homogeneous[1] / pixel_homogeneous[2]

    return u, v, Z_c

# new: 统计方块像素
def count_visible_cube_pixels(segmentation, cube_id):
    """
    由于flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
    因此segmentation保存两部分内容, 一部分为objectid, 另一部分为linkindex
    假如RGB图片某个像素[255, 0, 0], 那么 segmentation 可能直接为50331655,
    它是由objectid = 7, linkindex = 2算出, segmentation = objectid + (linkindex + 1)^24, 如果没有拍到物体自动置为-1
    """
    object_ids = segmentation & (1 << 24) - 1 # 左移24位减1之后就有24个1, & 是按位与, 也是可以把高位删除 低位保存即object_id
    cube_mask = (segmentation >= 0) & (object_ids == cube_id)
    visible_pixels = np.count_nonzero(cube_mask)
    return int(visible_pixels)

# new: 完整方块应该有多少像素
def get_clean_cube_pixel_count(robot_id, cube_id):
    old_position, old_orientation = p.getBasePositionAndOrientation(robot_id)
    p.resetBasePositionAndOrientation(robot_id, [-5.0, 0.0, TABLE_TOP_Z], old_orientation)
    for _ in range(3):
        p.stepSimulation()
    _, segmentation = capture_rgb_and_segmentation()
    clean_pixels = count_visible_cube_pixels(segmentation, cube_id)
    p.resetBasePositionAndOrientation(robot_id, old_position, old_orientation)
    return clean_pixels

# new: 随机设置机械臂姿态
def set_occluding_robot_pose(
    robot_id,
    joint_indices,
    end_effector_link,
    home_ee_orientation,
    cube_position
):
    theta = np.random.uniform(0.0, 2.0 * np.pi)
    radius = np.random.uniform(0.035, 0.060)
    dx = radius * np.cos(theta)
    dy = radius * np.sin(theta)
    target_position = [
        cube_position[0] + dx,
        cube_position[1] + dy,
        TABLE_TOP_Z + 2 * CUBE_HALF + np.random.uniform(0.1, 0.18)
    ]
    ik_solution = p.calculateInverseKinematics(
        robot_id,
        end_effector_link,
        targetPosition=target_position,
        targetOrientation=home_ee_orientation,
        maxNumIterations=200,
        residualThreshold=1e-5
    )
    for joint_index, joint_angle in zip(joint_indices, ik_solution):
        p.resetJointState(robot_id, joint_index, joint_angle)
    
    for _ in range(5):
        p.stepSimulation()

# new: 初始化最佳结果
def generate_occluded_rgb(
    robot_id,
    joint_indices,
    end_effector_link,
    home_ee_orientation,
    cube_id,
    cube_position,
    clean_pixels
):
    best_rgb = None
    best_occlusion_ratio = 0.0
    target_ratio = (MIN_OCCLUSION_RATIO + MAX_OCCLUSION_RATIO) / 2.0
    best_distance = np.inf
    for attempt in range(MAX_OCCLUSION_TRIES):
        set_occluding_robot_pose(
            robot_id,
            joint_indices,
            end_effector_link,
            home_ee_orientation,
            cube_position
        )

        rgb, segmentation = capture_rgb_and_segmentation()
        visible_pixels = count_visible_cube_pixels(segmentation, cube_id)
        clean_pixels = get_clean_cube_pixel_count(robot_id, cube_id)
        
        visible_ratio = visible_pixels / clean_pixels
        visible_ratio = np.clip(visible_ratio, 0.0, 1.0)
        occlusion_ratio = 1.0 - visible_ratio
        if MIN_OCCLUSION_RATIO <= occlusion_ratio <= MAX_OCCLUSION_RATIO:
            return rgb, float(occlusion_ratio)

        distance = abs(occlusion_ratio - target_ratio)
        if distance < best_distance:
            best_distance = distance
            best_rgb = rgb.copy()
            best_occlusion_ratio = occlusion_ratio
    return best_rgb, best_occlusion_ratio
    
# 11. generate dataset
# revise: 加入遮挡训练样本
def generate_dataset(
    robot_id,
    joint_indices,
    end_effector_link,
    home_ee_orientation,
    cube_id,
    R_cw,
    K,
    number_of_samples
):
    images = []
    targets = []
    occlusion_flags = [] # 遮挡标记
    occlusion_ratios = [] # 遮挡比例

    while len(images) < number_of_samples:
        cube_x = np.random.uniform(
            0.38,
            0.72
        )
        cube_y = np.random.uniform(
            -0.22,
            0.22
        )
        cube_z = TABLE_TOP_Z + CUBE_HALF

        cube_position = np.array([
            cube_x, cube_y, cube_z
        ], dtype=np.float64)

        p.resetBasePositionAndOrientation(
            cube_id,
            [
                cube_x,
                cube_y,
                cube_z
            ],
            [
                0,
                0,
                0,
                1
            ]
        )

        reset_home(robot_id, joint_indices)

        projection_result = world_to_pixel(cube_position, R_cw, K)

        if projection_result is None:
            continue

        u_center, v_center, Z_c = projection_result
        if not 0.0 <= u_center < WIDTH and 0.0 <= v_center < HEIGHT:
            continue

        
        clean_pixels = get_clean_cube_pixel_count(robot_id, cube_id)
        if clean_pixels <= 0:
            continue

        random_number = np.random.rand()
        use_occlusion = random_number < OCCLUSION_PROBABILITY
        if use_occlusion:
            rgb, occlusion_ratio = generate_occluded_rgb(
                robot_id,
                joint_indices,
                end_effector_link,
                home_ee_orientation,
                cube_id,
                cube_position,
                clean_pixels
            )
        else:
            reset_home(robot_id, joint_indices)
            rgb, _ = capture_rgb_and_segmentation()
            occlusion_ratio = 0.0
        
        target_heatmap = make_gaussian_heatmap(
            u_center,
            v_center,
            sigma=4.0
        )

        image = rgb.astype(
            np.float32
        )

        image = image / 255.0 # H W C
        image = np.transpose(
            image,
            (2, 0, 1)
        ) # C H W

        target_heatmap = np.expand_dims(
            target_heatmap,
            axis=0
        ) # 增加一个新维度(0, H, W)

        images.append(image)
        targets.append(target_heatmap)

        occlusion_flags.append(use_occlusion)
        occlusion_ratios.append(occlusion_ratio)
        
        print("generated:", len(images), "/", number_of_samples, \
              "center =", (round(u_center, 1), round(v_center, 1)), \
              "occlusion =", round(occlusion_ratio, 2))
    
    images = np.stack(
        images,
        axis=0
    )

    targets = np.stack(
        targets,
        axis=0
    )

    occlusion_flags = np.array(
        occlusion_flags,
        dtype=bool
    )

    occlusion_ratios = np.array(
        occlusion_ratios,
        dtype=np.float32
    )

    images = torch.from_numpy(images)
    targets = torch.from_numpy(targets)
    

    return images, targets, occlusion_flags, occlusion_ratios

def show_dataset_examples(
    train_x,
    train_y,
    train_occlusion_flags,
    train_occlusion_ratios):
    number_to_show = min(4, train_x.shape[0])
    plt.figure(figsize=(10, 2.8 * number_to_show))
    for i in range(number_to_show):
        image = train_x[i].numpy()
        image = np.transpose(image, (1, 2, 0))
        target = train_y[i, 0].numpy()
        plt.subplot(number_to_show, 2, 2 * i + 1)
        plt.imshow(image)
        plt.title("Occluded = " + str(train_occlusion_flags[i]) + "\nratio" + f"{train_occlusion_ratios[i]}")
        plt.axis("off")
        plt.subplot(number_to_show, 2, 2 * i + 2)
        plt.imshow(target, cmap="hot")
        plt.title("Ground Truth Heatmap")
        plt.axis("off")
        plt.colorbar()
    plt.tight_layout()
    plt.show()

# new
def show_prediction_examples(
    model,
    device,
    val_x,
    val_y,
    val_occlusion_flags
):
    occluded_indices = np.where(val_occlusion_flags)[0]
    if len(occluded_indices) > 0:
        sample_index = int(occluded_indices[0])
    else:
        sample_index = 0
    x = val_x[sample_index:sample_index + 1].to(device)
    model.eval()
    with torch.no_grad():
        prediction = model(x)
    predicted_heatmap = prediction[0, 0].cpu().numpy()
    true_heatmap = val_y[sample_index, 0].cpu().numpy()
    image = val_x[sample_index].cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    u_pred, v_pred = heatmap_to_pixel(predicted_heatmap)
    u_true, v_true = heatmap_to_pixel(true_heatmap)
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Input RGB")
    plt.scatter(u_true, v_true, marker="x", s=80)
    plt.scatter(u_pred, v_pred, marker="o", s=80)
    plt.subplot(1, 3, 2)
    plt.imshow(predicted_heatmap, cmap="hot")
    plt.title("Predicted heatmap")
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.imshow(true_heatmap, cmap="hot")
    plt.title("True heatmap")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

#12. build small CNN
class TinyHeatmapNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()

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

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.output_layer(x)
        x = self.sigmoid(x)

        return x

# 13. definite mse loss
def weight_mse_loss(prediction, target):

    weight = 1.0 + 10.0 * target
    squared_error = (prediction - target) ** 2
    weight_error = weight * squared_error

    return weight_error.mean()

# 14. heatmap to pixel
def heatmap_to_pixel(heatmap):
    flat_index = np.argmax(heatmap)

    v, u = np.unravel_index(
        flat_index,
        heatmap.shape
    )

    return int(u), int(v)

# 15. calculate pixel error
def calculate_pixel_error(
    prediction,
    target
):
    prediction_np = prediction.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    errors = []

    batch_size = prediction_np.shape[0]

    for sample_index in range(batch_size):
        predicted_heatmap = prediction_np[sample_index, 0]
        target_heatmap = target_np[sample_index, 0]

        u_pred, v_pred = heatmap_to_pixel(predicted_heatmap)
        u_true, v_true = heatmap_to_pixel(target_heatmap)

        pixel_error = np.sqrt((u_pred - u_true) ** 2 + (v_pred - v_true) ** 2)

        errors.append(pixel_error)

    return np.array(errors, dtype=np.float32)

# new: 保存平均值，单独写出来为了计算遮挡和不遮挡的误差
def safe_mean(value):
    if len(value) == 0:
        return NaN
    return np.mean(value)

# 16. train
# revised: 增加一些误差
def train_model(
    train_x,
    train_y,
    val_x,
    val_y,
    val_occlusion_flags
):
    if torch.cuda.is_available():
        device = torch.device(
            "cuda"
        )
    else:
        device = torch.device(
            "cpu"
        )
    
    model = TinyHeatmapNet()
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    train_x = train_x.to(device)
    train_y = train_y.to(device)
    val_x = val_x.to(device)
    val_y = val_y.to(device)

    for epoch in range(EPOCHS):
        model.train()

        permutation = torch.randperm(train_x.shape[0], device=device)

        epoch_losses = []

        for start in range(0, train_x.shape[0], BATCH_SIZE):
            batch_indices = permutation[start:start + BATCH_SIZE]
            x = train_x[batch_indices]
            y = train_y[batch_indices]

            prediction = model(x)
            loss = weight_mse_loss(prediction, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.item()))
        
        model.eval()

        with torch.no_grad():
            val_prediction = model(val_x)

        pixel_error = calculate_pixel_error(
            val_prediction,
            val_y
        )

        all_errors = safe_mean(pixel_error)
        clean_errors = safe_mean(pixel_error[~val_occlusion_flags])
        occluded_errors = safe_mean(pixel_error[val_occlusion_flags])

        mean_loss = np.mean(epoch_losses)

        print("Epoch", epoch + 1, "| Loss =", round(float(mean_loss), 6), \
              "\nAll errors =", round(float(all_errors), 2), "px", \
               "\nClean errors =", round(float(clean_errors), 2), "px", \
                "\nOccluded errors =", round(float(occluded_errors), 2), "px")

    return model, device

# 17. main
# revised: 增加画图函数
def main():
    np.random.seed(0)
    torch.manual_seed(0)

    R_cw = build_world_to_camera_rotation()
    K = build_camera_intrinsic_matrix()

    table_id, robot_id, cube_id, joint_indices, end_effector_link, home_ee_orientation = build_training_world()
    train_x, train_y, train_occlusion_flags, train_occlusion_ratios = generate_dataset(
        robot_id,
        joint_indices,
        end_effector_link,
        home_ee_orientation,
        cube_id,
        R_cw,
        K,
        TRAIN_SAMPLES
    )
    val_x, val_y, val_occlusion_flags, val_occlusion_ratios = generate_dataset(
        robot_id,
        joint_indices,
        end_effector_link,
        home_ee_orientation,
        cube_id,
        R_cw,
        K,
        VAL_SAMPLES
    )
    print("train_x shape:", train_x.shape, "\ntrain_y shape:", train_y.shape)

    show_dataset_examples(
        train_x,
        train_y,
        train_occlusion_flags,
        train_occlusion_ratios
    )

    model, device = train_model(
        train_x,
        train_y,
        val_x,
        val_y,
        val_occlusion_flags
    )

    show_prediction_examples(
            model,
            device,
            val_x,
            val_y,
            val_occlusion_flags
        )

    model = model.to("cpu")
    torch.save(model.state_dict(), MODEL_FILE)
    print("Model saved to:", MODEL_FILE)

    p.disconnect()

if __name__ == "__main__":
    main()