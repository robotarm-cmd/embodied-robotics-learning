"""
PyBullet 随机生成训练图片
        ↓
自动得到方块真实 Pixel
        ↓
制作 Heatmap 标签
        ↓
RGB → CNN → 预测 Heatmap
        ↓
Loss
        ↓
Backward
        ↓
修改 CNN 参数
        ↓
保存 cube_heatmap_model.pt


加载 cube_heatmap_model.pt
        ↓
Camera 拍一张新的 RGB-D
        ↓
RGB → 同一个 CNN → Heatmap
        ↓
Heatmap 最大位置 → (u,v)
        ↓
Depth[v,u]
        ↓
Pixel → Camera 3D
        ↓
Camera → World
        ↓
方块 XYZ
        ↓
视觉反馈
        ↓
IK
        ↓
UR5
"""

import numpy as np
import matplotlib.pyplot as plt

from numpy.random import weibull
from sympy import Plane
import torch
from torch.autograd.functional import _validate_v
from torch.cpu import is_available
import torch.nn as nn

import pybullet as p
import pybullet_data

from robot_descriptions.loaders.pybullet import load_robot_description

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

EPOCHS = 10

LEARNING_RATE = 0.001

MODEL_FILE = "cube_heatmap_model.pt"

# 5. UR5 initial degree
HOME_Q_DEG = np.array(
    [0.0, -90.0, 90.0, -90.0, -90.0, 0.0],
    dtype=float
)

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

# 8. build training world
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

    p.createMultiBody(
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

    joint_indices = get_arm_joints(robot_id)
    reset_home(
        robot_id,
        joint_indices
    )

    return cube_id

# 9. capture rgb and segmentation
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

    segmentation = np.asarray(
        image[4],
        dtype=np.int32
    )

    segmentation = segmentation.reshape(
        HEIGHT,
        WIDTH,
    )

    return rgb, segmentation

# 10. find the truth center pixel by cube_id and segmentation
def get_cube_center_from_segmentaion(
    segmentation,
    cube_id
):
    cube_mask = cube_id == segmentation
    rows, cols = np.where(
        cube_mask
    ) # v -> rows, u -> cols

    if len(rows) == 0:
        return None

    u = float(np.mean(cols))
    v = float(np.mean(rows))

    return u, v

# 11. make heatmap
def make_gaussian_heatmap(
    u_center,
    v_center,
    sigma=4.0
):
    x_coordinates = np.arange(WIDTH)
    y_coordinates = np.arange(HEIGHT)

    grid_x, grid_y = np.meshgrid(
        x_coordinates,
        y_coordinates
    )

    squared_distance = (grid_x - u_center)**2 + (grid_y - v_center)**2
    heat_map = np.exp(-squared_distance / (2 * sigma ** 2))

    return heat_map.astype(
        np.float32
    )

# 11. generate dataset
def generate_dataset(
    cube_id,
    number_of_samples
):
    images = []
    targets = []

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

        rgb, segmentation = capture_rgb_and_segmentation()

        center = (get_cube_center_from_segmentaion(
            segmentation,
            cube_id
        ))

        if center is None:
            continue
        
        u_center, v_center = center
        target_heatmap = make_gaussian_heatmap(
            u_center,
            v_center
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
        print("generated:", len(images), "/", number_of_samples)
    
    images = np.stack(
        images,
        axis=0
    )

    targets = np.stack(
        targets,
        axis=0
    )

    images = torch.from_numpy(images)
    targets = torch.from_numpy(targets)

    return images, targets

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

    return float(np.mean(errors))

# 16. train
def train_model(
    train_x,
    train_y,
    val_x,
    val_y
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

        mean_loss = np.mean(epoch_losses)

        print("Epoch", epoch + 1, "| Loss =", round(float(mean_loss), 6), \
              "| Val pixel error =", round(float(pixel_error), 2), "px")
    
    # 取验证集第 1 张图片
    true_heatmap = (
        val_y[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    pred_heatmap = (
        val_prediction[0, 0]
        .detach()
        .cpu()
        .numpy()
    )


    plt.figure(
        figsize=(10, 4)
    )


    # -----------------------------
    # 左边：真实热力图
    # -----------------------------
    plt.subplot(
        1,
        2,
        1
    )

    plt.imshow(
        true_heatmap,
        cmap="hot"
    )

    plt.colorbar()

    plt.title(
        "True Heatmap"
    )


    # -----------------------------
    # 右边：CNN预测热力图
    # -----------------------------
    plt.subplot(
        1,
        2,
        2
    )

    plt.imshow(
        pred_heatmap,
        cmap="hot"
    )

    plt.colorbar()

    plt.title(
        "Predicted Heatmap"
    )


    plt.tight_layout()

    plt.show()
    
    model = model.to("cpu")
    torch.save(model.state_dict(), MODEL_FILE)
    print("Model saved to:", MODEL_FILE)

    return model

# 17. main
def main():
    np.random.seed(0)
    torch.manual_seed(0)

    cube_id = build_training_world()
    train_x, train_y = generate_dataset(cube_id, TRAIN_SAMPLES)
    val_x, val_y = generate_dataset(cube_id, VAL_SAMPLES)
    print("train_x shape:", train_x.shape, "\ntrain_y shape:", train_y.shape)

    model = train_model(
        train_x,
        train_y,
        val_x,
        val_y
    )

    _ = model

    

    p.disconnect()


if __name__ == "__main__":
    main()












