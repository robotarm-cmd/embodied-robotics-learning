import math
import os

import numpy as np

import torch
from torch._dynamo.polyfills import NoEnterTorchFunctionMode
from torch.cuda import is_available
import torch.nn as nn

import matplotlib.pyplot as plt

import mujoco

WIDTH = 128
HEIGHT = 128

CAMERA_NAME = "top_camera"

TABLE_TOP_Z = 0.0

CUBE_HALF = 0.03
PLACE_HALF = 0.005

PATCH_SIZE = 16
EMBED_DIM = 128

NUMBER_OF_BLOCKS = 2

TRAIN_SAMPLES = 600
VAL_SAMPLES = 150

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

MIN_TARGET_DISTANCE = 0.16

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint"
]

ACTUATOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3"
]

HOME_Q_DEG = np.array([
    0.0,
    -90.0,
    90.0,
    -90.0,
    -90.0,
    0.0
], dtype=np.float64)

def load_world(xml_path):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    return model, data

def set_robot_home(model, data):
    home_q = np.deg2rad(HOME_Q_DEG)
    for i in range(6):
        joint_name = JOINT_NAMES[i]
        actuator_name = ACTUATOR_NAMES[i]
        
        data.joint(joint_name).qpos[0] = home_q[i]
        data.joint(joint_name).qvel[0] = 0.0

        actuator_id = model.actuator(actuator_name).id
        data.ctrl[actuator_id] = home_q[i]

    mujoco.mj_forward(model, data)

def get_camera_geometry(model, data):
    camera_id = model.camera(CAMERA_NAME).id

    camera_position = data.camera(
        CAMERA_NAME
    ).xpos.copy()

    R_mujoco = data.camera(
        CAMERA_NAME
    ).xmat.reshape(3, 3).copy()

    x_cw = R_mujoco[:, 0]
    y_cw = -R_mujoco[:, 1]
    z_cw = -R_mujoco[:, 2]

    R_wc = np.column_stack([
        x_cw,
        y_cw,
        z_cw
    ])

    R_cw = R_wc.T

    fov_y_deg = model.cam_fovy[camera_id]
    fov_y_rad = np.deg2rad(fov_y_deg)

    fy = HEIGHT / (
        2.0 * np.tan(fov_y_rad / 2.0)
    )

    fx = fy

    cx = (WIDTH - 1) / 2.0
    cy = (HEIGHT - 1) / 2.0

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    return camera_position, R_cw, K

def pixel_to_world_on_plane(u, v, plane_z, camera_position, R_cw, K):
    pixel = np.array([u, v, 1.0], dtype=np.float32)
    K_inv = np.linalg.inv(K)
    ray_camera = K_inv @ pixel
    R_wc = R_cw.T
    ray_world = R_wc @ ray_camera
    ray_z = ray_world[2]
    if abs(ray_z) < 1e-12:
        return None

    lambda_value = (plane_z - camera_position[2]) / ray_z
    if lambda_value <= 0.0:
        return None

    point_world = camera_position + lambda_value * ray_world

    return point_world

def build_robot_targets(pick_position, place_visual_position):
    pick_target = pick_position.copy() # cube center
    place_target = place_visual_position.copy()
    place_target[2] = TABLE_TOP_Z + CUBE_HALF

    pick_approach = pick_target.copy()
    pick_approach[2] += 0.1

    place_approach = place_target.copy()
    place_approach[2] += 0.1

    return pick_position, pick_approach, place_target, place_approach


def world_to_pixel(world_point, camera_position, R_cw, K):
    relative_position = world_point - camera_position
    P_c = R_cw @ relative_position
    homegeneous_pixel = K @ P_c
    if homegeneous_pixel[2] < 0:
        return None
    u = homegeneous_pixel[0] / homegeneous_pixel[2]
    v = homegeneous_pixel[1] / homegeneous_pixel[2]

    return u, v

def sample_task_position():
    while True:
        cube_x = np.random.uniform(0.38, 0.68)
        cube_y = np.random.uniform(-0.22, 0.22)

        place_x = np.random.uniform(0.19, 0.36)
        place_y = np.random.uniform(-0.19, 0.32)

        cube_position = np.array([cube_x, cube_y, CUBE_HALF + TABLE_TOP_Z], dtype=np.float32)
        place_position = np.array([place_x, place_y, PLACE_HALF + TABLE_TOP_Z], dtype=np.float32)

        xy_difference = cube_position[:2] - place_position[:2]
        xy_distance = np.linalg.norm(xy_difference)
        if xy_distance >= MIN_TARGET_DISTANCE:
            return cube_position, place_position

def set_task_position(model, data, cube_position, place_position):
    cube_body_id = model.body("cube").id
    place_body_id = model.body("place").id

    model.body_pos[cube_body_id] = cube_position
    model.body_pos[place_body_id] = place_position

    mujoco.mj_forward(model, data)

def capture_rgb(renderer, data):
    renderer.update_scene(data, camera=CAMERA_NAME)
    rgb = renderer.render().copy()

    return rgb # H W C

# 生成真正的mujoco dataset
# 随机cube/place位置 -> 放进mujoco -> 计算真实pixel label -> camera真正渲染rgb -> rgb / 255 -> 保存image -> 保存4维label
def generate_dataset(number_of_sample, model, data, renderer):
    images = []
    targets = []

    while len(images) < number_of_sample:
        cube_position, place_position = sample_task_position()
        set_task_position(model, data, cube_position, place_position)
        camera_position, R_cw, K = get_camera_geometry(model, data)
        cube_projection = world_to_pixel(cube_position, camera_position, R_cw, K)
        place_projection = world_to_pixel(place_position, camera_position, R_cw, K)
        if cube_projection is None or place_projection is None:
            continue
        pick_u, pick_v = cube_projection
        place_u, place_v = place_projection

        if pick_u < 0 or pick_u >= WIDTH:
            continue
        if pick_v < 0 or pick_v >= HEIGHT:
            continue
        if place_u < 0 or place_u >= WIDTH:
            continue
        if place_v < 0 or place_v >= HEIGHT:
            continue
        
        rgb = capture_rgb(renderer, data)
        image = rgb.astype(np.float32)
        image = image / 255.0
        images.append(image)
        target = np.array([
            pick_u / (WIDTH - 1),
            pick_v / (HEIGHT - 1),
            place_u / (WIDTH - 1),
            place_v / (HEIGHT - 1)
        ], dtype=np.float32)
        targets.append(target)
    
    images = np.stack(images, axis=0)
    targets = np.stack(targets, axis=0)
    
    images = images.transpose(0, 3, 1, 2) # B C H W

    images = torch.from_numpy(images)
    targets = torch.from_numpy(targets)

    return images, targets
    
class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(in_channels=3, out_channels=EMBED_DIM, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
    
    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(start_dim=2)
        x = x.transpose(2, 1)
    
        return x

class SingleHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        self.embed_dim = embed_dim
    
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attention = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.embed_dim), dim=-1)
        output = torch.matmul(attention, v)
        output = self.output_projection(output)

        return output, attention

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = SingleHeadSelfAttention(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, x):
        attention_input = self.norm1(x)
        attention_output, attention = self.attention(attention_input)
        x = x + attention_output
        mlp_input = self.norm2(x)
        mlp_output = self.mlp(mlp_input)
        x = x + mlp_output

        return x, attention

class TinyViTPickPlace(nn.Module):
    def __init__(self):
        super().__init__()

        patch_per_row = HEIGHT // PATCH_SIZE
        patch_per_col = WIDTH // PATCH_SIZE
        self.number_of_patches = patch_per_row * patch_per_col
        self.patch_embedding = PatchEmbedding()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))
        self.position_embedding = nn.Parameter(torch.zeros(1, self.number_of_patches + 1, EMBED_DIM))
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, 4)
        self.blocks = nn.ModuleList([
            TransformerBlock(EMBED_DIM) 
            for _ in range(NUMBER_OF_BLOCKS)
        ])

    def forward(self, image):
        x = self.patch_embedding(image)
        batch_size = x.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.position_embedding
        last_attention = None
        for block in self.blocks:
            x, last_attention = block(x)
        x = self.norm(x)
        cls_feature = x[:, 0]
        output = self.head(cls_feature)
        output = torch.sigmoid(output)

        return output, last_attention

def coordinate_regression_loss(prediction, target):
    squared_error = (prediction - target) ** 2
    loss = squared_error.mean()

    return loss

def pixel_error(prediction, target):

    prediction_pixel = prediction.clone()
    target_pixel = target.clone()

    prediction_pixel[:, 0] *= WIDTH - 1
    prediction_pixel[:, 1] *= HEIGHT - 1
    prediction_pixel[:, 2] *= WIDTH - 1
    prediction_pixel[:, 3] *= HEIGHT - 1

    target_pixel[:, 0] *= WIDTH - 1
    target_pixel[:, 1] *= HEIGHT - 1
    target_pixel[:, 2] *= WIDTH - 1
    target_pixel[:, 3] *= HEIGHT - 1


    pick_difference = prediction_pixel[:, 0:2] - target_pixel[:, 0:2]
    place_difference = prediction_pixel[:, 2:4] - target_pixel[:, 2:4]
    pick_distance = torch.sqrt(torch.sum(pick_difference ** 2, dim=1))
    place_distance = torch.sqrt(torch.sum(place_difference ** 2, dim=1))

    pick_error = float(pick_distance.mean().item())
    place_error = float(place_distance.mean().item())

    return pick_error, place_error

def train_model(train_x, train_y, val_x, val_y):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = TinyViTPickPlace()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_x = train_x.to(device)
    train_y = train_y.to(device)
    val_x = val_x.to(device)
    val_y = val_y.to(device)

    

    for epoch in range(EPOCHS):
        model.train()
        permutation = torch.randperm(train_x.shape[0], device=device)
        epoch_loss = []

        for start in range(0, train_x.shape[0], BATCH_SIZE):
            image_indices = permutation[start : start + BATCH_SIZE]
            images = train_x[image_indices]
            targets = train_y[image_indices]
            predicted_ouput, attention_output = model(images)
            loss = coordinate_regression_loss(predicted_ouput, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            predicted_val, _ = model(val_x)
            pick_error, place_error = pixel_error(predicted_val, val_y)
        
        print("Epoch =", epoch, "Loss =", round(float(np.mean(epoch_loss)), 6), "Pick error =", round(pick_error, 2), "Place error =", round(place_error, 2))

    model.to("cpu")

    return model

def predict_pick_place_pixels(model, rgb):
    image = rgb.astype(np.float32)
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    input_tensor = torch.from_numpy(image)
    input_tensor = input_tensor.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        prediction, attention = model(input_tensor)

    pick_u = prediction[0, 0].item() * (WIDTH - 1)
    pick_v = prediction[0, 1].item() * (HEIGHT - 1)
    place_u = prediction[0, 2].item() * (WIDTH - 1)
    place_v = prediction[0, 3].item() * (HEIGHT - 1)

    return pick_u, pick_v, place_u, place_v, attention

def show_result(rgb, true_pick_pixel, true_place_pixel, predicted_pick_pixel, predicted_place_pixel,attention):
    cls_attention = attention[0, 0, 1:]
    patches_per_row = WIDTH // PATCH_SIZE
    patches_per_col = HEIGHT // PATCH_SIZE
    attention_map = cls_attention.reshape(patches_per_row, patches_per_col).numpy()
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb)
    plt.scatter(true_pick_pixel[0], true_pick_pixel[1], marker="x", s=100, label="True Pick")
    plt.scatter(predicted_pick_pixel[0], predicted_pick_pixel[1], marker="^", s=100, label="Predicted Pick")
    plt.scatter(true_place_pixel[0], true_place_pixel[1], marker="x", s=100, label="True Place")
    plt.scatter(predicted_place_pixel[0], predicted_place_pixel[1], marker="^", s=100, label="Predicted Place")
    plt.legend()
    plt.title("MuJoCo RGB")
    plt.subplot(1, 2, 2)
    plt.imshow(attention_map)
    plt.colorbar()
    plt.title("CLS Attention")
    plt.tight_layout()
    plt.show()

def main():
    np.random.seed(1)
    torch.manual_seed(1)

    # 1. XML path
    file_name = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(file_name, "mujoco_menagerie", "universal_robots_ur5e", "ur5e_scene.xml")
    
    # 2. Load mujoco
    model, data = load_world(xml_path)

    # 3. Set robot HOME
    set_robot_home(model, data)

    # 4. Camera Geormetry
    camera_position, R_cw, K = get_camera_geometry(model, data)

    # 5. Create renderer
    with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
        print("\nGnerating training dataset ...")

        # 6. Generate train dataset
        train_x, train_y = generate_dataset(TRAIN_SAMPLES, model, data, renderer)
        print("Train images:", train_x.shape)
        print("Train targets:", train_y.shape)

        # 7. Generate val dataset
        val_x, val_y = generate_dataset(VAL_SAMPLES, model, data, renderer)
        print("Validation images:", val_x.shape)
        print("Validation targets:", val_y.shape)

        # 8. Train ViT
        vit = train_model(train_x, train_y, val_x, val_y)

        # 9. Save train model
        model_file = os.path.join(file_name, "vit_pick_place.pt")
        torch.save(vit.state_dict(), model_file)
        print("\nSaved model:", model_file)

        # 10. Create new test scene
        true_cube_position, true_place_position = sample_task_position()
        set_task_position(model, data, true_cube_position, true_place_position)

        # 11. Capture test rgb
        test_rgb = capture_rgb(renderer, data)

        # 12. Ground truth pixel
        true_pick_pixel = world_to_pixel(true_cube_position, camera_position, R_cw, K)
        true_place_pixel = world_to_pixel(true_place_position, camera_position, R_cw, K)

        # 13. ViT predict pixels
        pick_u, pick_v, place_u, place_v, attention = predict_pick_place_pixels(vit, test_rgb)
        predicted_pick_pixel = (pick_u, pick_v)
        predicted_place_pixel = (place_u, place_v)

        # 14. Predicted pixels -> world
        pick_plane_z = TABLE_TOP_Z + CUBE_HALF
        place_plane_z = TABLE_TOP_Z + PLACE_HALF
        predicted_pick_position = pixel_to_world_on_plane(pick_u, pick_v, pick_plane_z, camera_position, R_cw, K)
        predicted_place_position = pixel_to_world_on_plane(place_u, place_v, place_plane_z, camera_position, R_cw, K)

        print("Predicted pick position =", predicted_pick_position)
        print("Predicted place position =", predicted_place_position)

        # 15. Build robot cartesian targets
        pick_target, pick_approach, place_target, place_approach = build_robot_targets(
            predicted_pick_position,
            predicted_place_position
        )

        # 16. Evaluation
        pick_world_error = np.linalg.norm(predicted_pick_position - true_cube_position)
        place_world_error = np.linalg.norm(predicted_place_position - true_place_position)

        print("\nPick world error:", pick_world_error)
        print("\nPlace world error:", place_world_error)

        # 17. Visualization
        show_result(test_rgb, true_pick_pixel, true_place_pixel, predicted_pick_pixel, predicted_place_pixel, attention)

if __name__ == "__main__":
    main()











