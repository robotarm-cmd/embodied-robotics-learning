import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import pybullet_data
from robot_descriptions.loaders.pybullet import load_robot_description
import torch.nn as nn
import torch

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

TRAIN_SAMPLES =200
VAL_SAMPLES = 40

BATCH_SIZE = 16

EPOCHS = 20

LEARNING_RATE = 0.001

OCCLUSION_PROBABILITY = 0.50

MIN_OCCLUSION_RATIO = 0.20
MAX_OCCLUSION_RATIO = 0.70

MAX_OCCLUSION_TRIES = 30

PICK_OCCLUSION_PROBILITY = 0.5

HOME_Q_DEG = np.array([0.0, -90.0, 90.0, -90.0, -90.0, 0.0], dtype=float)

MODEL_FILE = "pick_place_unet.pt"

def get_arm_joints(robot_id):
    joint_indices = []
    number_of_joints = p.getNumJoints(robot_id)

    for joint_index in range(number_of_joints):
        info = p.getJointInfo(robot_id, joint_index)

        if info[2] == p.JOINT_REVOLUTE:
            joint_indices.append(joint_index)
    if len(joint_indices) < 6:
        raise ValueError("没有找到UR5的6个旋转关节")
    
    return joint_indices[:6]

def reset_home(robot_id, joint_indices):
    q_home = np.deg2rad(HOME_Q_DEG)

    for joint_index, joint_angle in zip(joint_indices, q_home):
        p.resetJointState(robot_id, joint_index, float(joint_angle))

    for _ in range(2):
        p.stepSimulation()

def get_arm_ee(robot_id):
    candidate_names = ["tool0", "ee_link", "wrist_3_link"]
    number_of_joints = p.getNumJoints(robot_id)

    for joint_index in range(number_of_joints):
        info = p.getJointInfo(robot_id, joint_index)
        link_name = info[12].decode("utf-8")
        if link_name in candidate_names:
            return joint_index
    
    raise ValueError("没有找到UR5末端关节")
 
def build_world():
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0.0, 0.0, -9.81)
    p.setTimeStep(1.0 / 240.0)
    p.loadURDF("plane.urdf")
    

    table_half = [0.4, 0.32, TABLE_TOP_Z / 2.0]
    table_center = [0.55, 0.0, TABLE_TOP_Z / 2.0]
    table_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=table_center,
        rgbaColor=[0.72, 0.62, 0.50, 1.0]
    )
    p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=table_visual,
        basePosition=table_center
    )

    cube_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[CUBE_HALF, CUBE_HALF, CUBE_HALF],
        rgbaColor=[0.05, 0.20, 1.0, 1.0]
    )
    cube_id = p.createMultiBody(
        baseMass=1.0,
        baseVisualShapeIndex=cube_visual,
        basePosition=[
            0.55,
            0.0,
            TABLE_TOP_Z + CUBE_HALF
        ]
    )

    place_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[PLACE_HALF, PLACE_HALF, PLACE_HALF],
        rgbaColor=[0.05, 1.0, 0.10, 1.0]
    )
    place_id = p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=place_visual,
        basePosition=[
            0.15,
            0.35,
            TABLE_TOP_Z + PLACE_HALF]
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
    reset_home(robot_id, joint_indices)
    ee_link_index = get_arm_ee(robot_id)
    ee_state = p.getLinkState(robot_id, ee_link_index, computeForwardKinematics=True)
    ee_orientation = ee_state[5]

    return robot_id, cube_id, place_id, joint_indices, ee_link_index, ee_orientation

def capture_rgb_and_segmentation():
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
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
    )

    rgba = np.asarray(image[2], dtype=np.uint8)
    rgba = rgba.reshape(HEIGHT, WIDTH, 4)
    rgb = rgba[:, :, :3]

    segmentation = np.asarray(image[4], dtype=np.int32)
    segmentation = segmentation.reshape(HEIGHT, WIDTH)

    return rgb, segmentation

def make_gaussian_heatmap(u_center, v_center, sigma=4.0):
    u_coordinates = np.arange(WIDTH)
    v_coordinates = np.arange(HEIGHT)
    grid_u, grid_v = np.meshgrid(u_coordinates, v_coordinates)
    
    squared_distance = (grid_u - u_center) ** 2 + (grid_v - v_center) ** 2
    heat_map = np.exp(-squared_distance / (2 * sigma ** 2))

    return heat_map

def get_rotation_matrix():

    # 相机 +Z：从相机指向目标
    z_cw = CAMERA_TARGET - CAMERA_EYE
    z_cw = z_cw / np.linalg.norm(z_cw)

    # 相机 +X：图像向右
    x_cw = np.cross(
        z_cw,
        CAMERA_UP
    )
    x_cw = x_cw / np.linalg.norm(x_cw)

    # 相机 +Y：图像向下
    y_cw = np.cross(
        z_cw,
        x_cw
    )
    y_cw = y_cw / np.linalg.norm(y_cw)

    R_cw = np.vstack([
        x_cw,
        y_cw,
        z_cw
    ])

    return R_cw

def get_camera_intrinsic_matrix():
    fy = HEIGHT / (2 * np.tan(np.deg2rad(FOV_Y) / 2.0))
    fx = fy
    cx = (WIDTH - 1) / 2.0
    cy = (HEIGHT - 1) / 2.0
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])

    return K

def world_to_pixel(world_position, R_cw, K):
    position_relative_world  = world_position - CAMERA_EYE
    camera_position  = R_cw @ position_relative_world
    Z_c = camera_position[2]
    if Z_c <= 0:
        return None

    pixel_homogeneous = K @ camera_position
    u = pixel_homogeneous[0] / pixel_homogeneous[2]
    v = pixel_homogeneous[1] / pixel_homogeneous[2]

    return u, v, Z_c

def count_visible_pixels(segmentation, object_id):
    object_ids = segmentation & (1 << 24) - 1
    object_mask = (segmentation >= 0) & (object_ids == object_id)
    visible_pixels = np.count_nonzero(object_mask)
    return int(visible_pixels)

def get_clean_pixel_count(robot_id, object_id):
    old_position, old_orientation = p.getBasePositionAndOrientation(robot_id)
    p.resetBasePositionAndOrientation(robot_id, [-5.0, 0.0, TABLE_TOP_Z], old_orientation)
    for _ in range(3):
        p.stepSimulation()
    _, segmentation = capture_rgb_and_segmentation()
    clean_pixels = count_visible_pixels(segmentation, object_id)
    p.resetBasePositionAndOrientation(robot_id, old_position, old_orientation)
    for _ in range(3):
        p.stepSimulation()
        
    return clean_pixels

def set_occluding_robot_pose(
        robot_id,
        joint_indices,
        end_effector_link,
        target_position
):
    direction = CAMERA_EYE - target_position
    alpha = np.random.uniform(0.10, 0.22)
    ee_target_position = target_position + alpha * direction
    ee_target_position[0] += np.random.uniform(-0.015, 0.015)
    ee_target_position[1] += np.random.uniform(-0.015, 0.015)

    ik_solution = p.calculateInverseKinematics(
        robot_id,
        end_effector_link,
        targetPosition=ee_target_position.tolist(),
        maxNumIterations=200,
        residualThreshold=1e-5
    )
    for joint_index, joint_angle in zip(joint_indices, ik_solution):
        p.resetJointState(robot_id, joint_index, float(joint_angle))

    for _ in range(5):
        p.stepSimulation()

def generate_occluded_rgb(
        robot_id,
        joint_indices,
        end_effector_link,
        home_ee_orientation,
        target_id,
        target_position,
        clean_pixels
):
    best_rgb = None
    best_occlusion_ratio = 0.0
    best_distance = np.inf

    target_ratio = (MIN_OCCLUSION_RATIO + MAX_OCCLUSION_RATIO) / 2.0
    
    for attempt in range(MAX_OCCLUSION_TRIES):
        reset_home(robot_id, joint_indices)

        set_occluding_robot_pose(
            robot_id,
            joint_indices,
            end_effector_link,
            target_position
        )

        rgb, segmentation = capture_rgb_and_segmentation()
        visible_pixels = count_visible_pixels(segmentation, target_id)
        visible_ratio = visible_pixels / clean_pixels
        visible_ratio = np.clip(visible_ratio, 0.0, 1.0)
        occlusion_ratio = 1 - visible_ratio

        if MIN_OCCLUSION_RATIO <= occlusion_ratio <= MAX_OCCLUSION_RATIO:
            return rgb, float(occlusion_ratio)

        distance = abs(occlusion_ratio - target_ratio)
        if distance < best_distance:
            best_distance = distance
            best_rgb = rgb.copy()
            best_occlusion_ratio = occlusion_ratio

    return best_rgb, float(best_occlusion_ratio)

def generate_dataset(
    robot_id,
    joint_indices,
    end_effector_link,
    home_ee_orientation,
    cube_id,
    place_id,
    R_cw,
    K,
    number_of_samples
):
    images = []
    targets = []
    occlusion_types = [] # 0 无遮挡 1 主动遮挡 pick 2 主动遮挡 place
    occlusion_ratios = []

    while len(images) < number_of_samples:
        cube_x = np.random.uniform(0.38,0.72)
        cube_y = np.random.uniform(-0.22,0.22)
        cube_z = TABLE_TOP_Z + CUBE_HALF
        cube_position = np.array([cube_x, cube_y, cube_z], dtype=np.float32)
        place_x = np.random.uniform(0.19,0.36)
        place_y = np.random.uniform(-0.19, 0.32)
        place_z = TABLE_TOP_Z + PLACE_HALF
        place_position = np.array([place_x, place_y, place_z], dtype=np.float32)

        distance = np.sqrt((place_x - cube_x) ** 2 + (place_y - cube_y) ** 2)
        if distance < 0.16:
            continue

        p.resetBasePositionAndOrientation(cube_id, cube_position.tolist(), [0.0, 0.0, 0.0, 1.0])
        p.resetBasePositionAndOrientation(place_id, place_position.tolist(), [0.0, 0.0, 0.0, 1.0])

        reset_home(robot_id, joint_indices)

        projection_cube = world_to_pixel(cube_position, R_cw, K)
        projection_place = world_to_pixel(place_position, R_cw, K)

        if projection_cube is None or projection_place is None:
            continue

        cube_u_center, cube_v_center, _ = projection_cube
        place_u_center, place_v_center, _ = projection_place

        cube_inside = 0.0 <= cube_u_center <= WIDTH and 0.0 <= cube_v_center <= HEIGHT
        place_inside = 0.0 <= place_u_center <= WIDTH and 0.0 <= place_v_center <= HEIGHT

        if not cube_inside or not place_inside:
            continue

        clean_cube_pixels = get_clean_pixel_count(robot_id, cube_id)
        if clean_cube_pixels <= 0:
            continue

        clean_place_pixels = get_clean_pixel_count(robot_id, place_id)
        if clean_place_pixels <=0:
            continue

        random_number = np.random.rand()
        use_occlusion = random_number < OCCLUSION_PROBABILITY
        if use_occlusion:

            if np.random.rand() < 0.5:
                target_id = cube_id
                target_position = cube_position
                target_clean_pixels = clean_cube_pixels
                occlusion_type = 1

            else:
                target_id = place_id
                target_position = place_position
                target_clean_pixels = clean_place_pixels
                occlusion_type = 2

            rgb, occlusion_ratio = generate_occluded_rgb(
                robot_id,
                joint_indices,
                end_effector_link,
                home_ee_orientation,
                target_id,
                target_position,
                target_clean_pixels
            )
        
        else:
            reset_home(robot_id, joint_indices)
            rgb, _ = capture_rgb_and_segmentation()

            occlusion_type = 0
            occlusion_ratio = 0.0

        target_cube_heatmap = make_gaussian_heatmap(cube_u_center, cube_v_center)
        target_place_heatmap = make_gaussian_heatmap(place_u_center, place_v_center)
        target_heatmap = np.stack((target_cube_heatmap, target_place_heatmap), axis=0)

        image = rgb.astype(np.float32)

        image = image / 255.0  # H W C
        image = np.transpose(image,(2, 0, 1))  # C H W

        images.append(image)
        targets.append(target_heatmap)

        occlusion_ratios.append(occlusion_ratio)
        occlusion_types.append(occlusion_type)

        print("generated:", len(images), "/", number_of_samples, \
              "occlusion_type =", occlusion_type, \
              "occlusion_ratio =", round(occlusion_ratio, 2))

    images = np.stack(
        images,
        axis=0
    )

    targets = np.stack(
        targets,
        axis=0
    )

    occlusion_types = np.array(
        occlusion_types,
        dtype=np.int32
    )

    occlusion_ratios = np.array(
        occlusion_ratios,
        dtype=np.float32
    )

    images = torch.from_numpy(images)
    targets = torch.from_numpy(targets)

    return images, targets, occlusion_types, occlusion_ratios

def show_dataset_examples(
    train_x,
    train_y,
    train_occlusion_types,
    train_occlusion_ratios):
    number_to_show = min(4, train_x.shape[0])
    type_name = ["None", "Pick", "Place"]
    plt.figure(figsize=(10, 2.8 * number_to_show))
    for i in range(number_to_show):
        image = train_x[i].numpy()
        image = np.transpose(image, (1, 2, 0))
        pick_heatmap = train_y[i, 0].numpy()
        place_heatmap = train_y[i, 1].numpy()
        plt.subplot(number_to_show, 3, 3 * i + 1)
        plt.imshow(image)
        plt.title("Occluded = " + type_name[train_occlusion_types[i]] + "\nratio" + f"{train_occlusion_ratios[i]}")
        plt.axis("off")
        plt.subplot(number_to_show, 3, 3 * i + 2)
        plt.imshow(pick_heatmap, cmap="hot")
        plt.title("Ground Truth Pick Heatmap")
        plt.axis("off")
        plt.subplot(number_to_show, 3, 3 * i + 3)
        plt.imshow(place_heatmap, cmap="hot")
        plt.title("Ground Truth Place Heatmap")
        plt.axis("off")
        plt.colorbar()
    plt.tight_layout()
    plt.show()

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

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))

        return x

class SmallUnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = ConvBlock(3, 16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = ConvBlock(16, 32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bottleneck = ConvBlock(32, 64)

        self.up2 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=2,
            stride=2
        )
        self.dec2 = ConvBlock(64, 32)

        self.up1 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=16,
            kernel_size=2,
            stride=2
        )

        self.dec1 = ConvBlock(32, 16)
        self.output_layer = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        b = self.bottleneck(p2)
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        output = self.output_layer(d1)
        return output

def weight_mse_loss(prediction, target):

    weight = 1.0 + 10.0 * target
    squared_error = (prediction - target) ** 2
    weight_error = weight * squared_error

    return weight_error.mean()

def heatmap_to_pixel(heatmap):
    flat_index = np.argmax(heatmap)

    v, u = np.unravel_index(
        flat_index,
        heatmap.shape
    )

    return int(u), int(v)

def calculate_pixel_errors(prediction, target):
    prediction_np = prediction.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    pick_errors = []
    place_errors = []

    batch_size = prediction_np.shape[0]

    for sample_index in range(batch_size):
        predicted_pick = prediction_np[sample_index, 0]
        target_pick = target_np[sample_index, 0]
        
        u_pred, v_pred = heatmap_to_pixel(predicted_pick)
        u_true, v_true = heatmap_to_pixel(target_pick)

        pick_error = np.sqrt((u_pred - u_true)**2 + (v_pred - v_true)**2)
        pick_errors.append(pick_error)

        prediction_pixel = prediction_np[sample_index, 1]
        target_pixel = target_np[sample_index, 1]

        u_pred, v_pred = heatmap_to_pixel(prediction_pixel)
        u_true, v_true = heatmap_to_pixel(target_pixel)

        place_error = np.sqrt((u_pred - u_true)**2 + (v_pred - v_true)**2)
        place_errors.append(place_error)

    return float(np.mean(pick_errors)), float(np.mean(place_errors))


def train_model(
    train_x,
    train_y,
    val_x,
    val_y
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    model = SmallUnet()
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    val_x = val_x.to(device)
    val_y = val_y.to(device)
    num = train_x[0]

    for epoch in range(EPOCHS):
        model.train()
        permutation = torch.randperm(train_x.shape[0], device=device)
        epoch_losses = []

        for start in range(0, train_x.shape[0], BATCH_SIZE):
            batch_indices = permutation[start:start+BATCH_SIZE]
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
            val_prediction = torch.sigmoid(val_prediction)

        pick_error, place_error = calculate_pixel_errors(val_prediction, val_y)
        mean_loss = np.mean(epoch_losses)
        print("Epoch", epoch + 1, "Loss =", round(float(mean_loss), 6), "Pick error =", \
               round(pick_error, 2), "px", "Place error =", round(place_error, 2), "px")
    
    model = model.to("cpu")
    torch.save(model.state_dict(), MODEL_FILE)
    print("Model saved:", MODEL_FILE)

    return model

def show_prediction(
    model,
    val_x,
    val_y,
    number_to_show=4
):
    model.eval()
    number_to_show = min(number_to_show, val_x.shape[0])
    with torch.no_grad():
        prediction = model(val_x[:number_to_show])
    prediction = prediction.cpu().numpy()
    
    plt.figure(figsize=(12, 3 * number_to_show))

    for i in range(number_to_show):
        image = val_x[i].numpy()
        image = np.transpose(image, (1, 2, 0))
        predicted_pick = prediction[i, 0]
        true_pick = val_y[i, 0]
        pick_u_pred, pick_v_pred = heatmap_to_pixel(predicted_pick)
        pick_u_true, pick_v_true = heatmap_to_pixel(true_pick)

        predicted_place = prediction[i, 1]
        true_place = val_y[i, 1]
        place_u_pred, place_v_pred = heatmap_to_pixel(predicted_place)
        place_u_true, place_v_true = heatmap_to_pixel(true_place)

        plt.subplot(
            number_to_show,
            3,
            3 * i + 1
        )

        plt.imshow(image)

        plt.scatter(
            pick_u_true,
            pick_v_true,
            marker="x",
            s=80
        )

        plt.scatter(
            pick_u_pred,
            pick_v_pred,
            marker="o",
            s=60
        )

        plt.scatter(
            place_u_true,
            place_v_true,
            marker="x",
            s=80
        )

        plt.scatter(
            place_u_pred,
            place_v_pred,
            marker="o",
            s=60
        )

        plt.title("RGB")

        plt.axis("off")

        plt.subplot(
            number_to_show,
            3,
            3 * i + 2
        )

        plt.imshow(
            predicted_pick,
            cmap="hot"
        )

        plt.scatter(
            pick_u_true,
            pick_v_true,
            marker="x",
            s=80
        )

        plt.scatter(
            pick_u_pred,
            pick_v_pred,
            marker="o",
            s=60
        )

        plt.title(
            "Predicted Pick"
        )

        plt.subplot(
            number_to_show,
            3,
            3 * i + 3
        )

        plt.imshow(
            predicted_place,
            cmap="hot"
        )

        plt.scatter(
            place_u_true,
            place_v_true,
            marker="x",
            s=80
        )

        plt.scatter(
            place_u_pred,
            place_v_pred,
            marker="o",
            s=60
        )

        plt.title(
            "Predicted Place"
        )


    plt.tight_layout()

    plt.show()


def main():
    np.random.seed(1)
    torch.manual_seed(1)

    R_cw = get_rotation_matrix()
    K = get_camera_intrinsic_matrix()

    robot_id, cube_id, place_id, joint_indices, ee_link_index, ee_orientation = build_world()
    train_x, train_y, train_occlusion_types, train_occlusion_ratios = generate_dataset(
        robot_id,
        joint_indices,
        ee_link_index,
        ee_orientation,
        cube_id,
        place_id,
        R_cw,
        K,
        TRAIN_SAMPLES
    )

    val_x, val_y, val_occlusion_flags, val_occlusion_ratios = generate_dataset(
        robot_id,
        joint_indices,
        ee_link_index,
        ee_orientation,
        cube_id,
        place_id,
        R_cw,
        K,
        VAL_SAMPLES
    )

    show_dataset_examples(train_x, train_y, train_occlusion_types, train_occlusion_ratios)

    model = train_model(
        train_x,
        train_y,
        val_x,
        val_y
    )

    show_prediction(model, val_x, val_y, number_to_show=4)

    p.disconnect()

if __name__ == "__main__":
    main()