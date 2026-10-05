"""
object: RGB -> Pick/Place Pixel -> Pick/Place 3D
"""

import os
import numpy as np
import copy
import matplotlib.pyplot as plt
import math
import mujoco
import torch
import torch.nn as nn

HEIGHT = 128
WIDTH = 128
PATCH_SIZE = 32
EMBED_DIM = 128
NUMBER_OF_HEADS = 4
NUMBER_OF_BLOCKS = 3
TRAIN_SAMPLES = 2000
VAL_SAMPLES = 400
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
RUN_TINY_OVERFIT_TEST = True

TABLE_TOP_Z = 0.0
CUBE_HALF = 0.03
PLACE_HALF = 0.005
MIN_TARGET_DISTANCE = 0.16

TINY_OVERFIT_SAMPLES = 16
TINY_OVERFIT_STEPS = 1000
MIN_IMPROVEMENT = 0.1
EARLY_STOPPING_PATIENCE = 20

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

CAMERA_NAME = "top_camera"

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
        data.joint(JOINT_NAMES[i]).qpos[0] = home_q[i]
        data.joint(JOINT_NAMES[i]).qvel[0] = 0.0
        actuator_id = model.actuator(ACTUATOR_NAMES[i]).id
        data.ctrl[actuator_id] = home_q[i]

    mujoco.mj_forward(model, data)

def get_camera_geometry(model, data):
    camera_id = model.camera(CAMERA_NAME).id
    camera_position = data.camera(CAMERA_NAME).xpos.copy()
    R_mujoco = data.camera(CAMERA_NAME).xmat.reshape(3, 3).copy()
    x_axis_world = R_mujoco[:, 0]
    y_axis_world = -R_mujoco[:, 1]
    z_axis_world = -R_mujoco[:, 2]
    R_wc = np.column_stack([x_axis_world, y_axis_world, z_axis_world])
    R_cw = R_wc.T

    fov_y_deg = model.cam_fovy[camera_id]
    fov_y_rad = np.deg2rad(fov_y_deg)
    fy = HEIGHT / (2.0 * np.tan(fov_y_rad / 2.0))
    fx = fy
    cx = (WIDTH - 1) / 2.0
    cy = (HEIGHT - 1) / 2.0
    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ])

    return camera_position, R_cw, K

def world_to_pixel(world_position, camera_position, R_cw, K):
    camera_point = R_cw @ (world_position - camera_position)
    Z_c = camera_point[2]
    if Z_c <= 0:
        return None

    homegeneous_pixel = K @ camera_point
    u = homegeneous_pixel[0] / homegeneous_pixel[2]
    v = homegeneous_pixel[1] / homegeneous_pixel[2]

    return u, v

def sample_task_positions():
    while True:
        cube_x = np.random.uniform(0.38, 0.68)
        cube_y = np.random.uniform(-0.22, 0.22)
        place_x = np.random.uniform(0.19, 0.36)
        place_y = np.random.uniform(-0.19, 0.32)
        cube_position = np.array([cube_x, cube_y, TABLE_TOP_Z + CUBE_HALF], dtype=np.float32)
        place_position = np.array([place_x, place_y, TABLE_TOP_Z + PLACE_HALF], dtype=np.float32)
        xy_difference = cube_position[:2] - place_position[:2]
        xy_distance = np.linalg.norm(xy_difference)
        if xy_distance >= MIN_TARGET_DISTANCE:
            return cube_position, place_position

def set_task_positions(model, data, cube_position, place_position):
    cube_body_id = model.body("cube").id
    place_body_id = model.body("place").id
    model.body_pos[cube_body_id] = cube_position
    model.body_pos[place_body_id] = place_position
    mujoco.mj_forward(model, data)

def capture_rgb(renderer, data):
    renderer.update_scene(data, camera=CAMERA_NAME)
    rgb = renderer.render().copy()

    return rgb


def generate_dataset(number_of_samples, model, data, renderer, camera_position, R_cw, K):
    images = []
    targets = []

    while len(images) < number_of_samples:
        cube_position, place_position = sample_task_positions()
        projection_pick_pixel = world_to_pixel(cube_position, camera_position, R_cw, K)
        projection_place_pixel = world_to_pixel(place_position, camera_position, R_cw, K)
        if projection_pick_pixel is None:
            continue
        if projection_place_pixel is None:
            continue
        u_pick, v_pick = projection_pick_pixel
        u_place, v_place = projection_place_pixel
        if u_pick < 0 or u_pick > WIDTH:
            continue
        if u_place < 0 or u_place > WIDTH:
            continue
        if v_pick < 0 or v_pick > HEIGHT:
            continue
        if v_place < 0 or v_place > HEIGHT:
            continue

        set_task_positions(model, data, cube_position, place_position)
        image = capture_rgb(renderer, data)
        image = image.astype(np.float32)
        image = image / 255.0
        target = np.array([u_pick / (WIDTH - 1), v_pick / (HEIGHT - 1), u_place / (WIDTH - 1), v_place / (HEIGHT - 1)], dtype=np.float32)
        
        images.append(image)
        targets.append(target)
    
    images = np.stack(images, axis=0)
    targets = np.stack(targets, axis=0)
    images = np.transpose(images, (0, 3, 1, 2))

    images = torch.from_numpy(images)
    targets = torch.from_numpy(targets)

    return images, targets

def show_dataset_sample(image, target):
    image_numpy = image.permute(1, 2, 0).numpy()
    pick_u = target[0].item() * (WIDTH - 1)
    pick_v = target[1].item() * (HEIGHT - 1)
    place_u = target[2].item() * (WIDTH - 1)
    place_v = target[3].item() * (HEIGHT - 1)

    plt.figure(figsize=(6, 6))
    plt.imshow(image_numpy)
    plt.scatter(pick_u, pick_v, marker="x", s=120, label="True Pick")
    plt.scatter(place_u, place_v, marker="x", s=120, label="True Place")
    plt.legend()
    plt.title("Dataset Label Check")

    plt.show()

class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(in_channels=3, out_channels=EMBED_DIM, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)

    def forward(self, image):
        x = self.projection(image)
        x = torch.flatten(x, start_dim=2)
        x = x.transpose(1, 2)

        return x # B 256 64

class MultiheadSelfAttention(nn.Module):
    def __init__(self, embed_dim, number_of_heads):
        super().__init__()

        if embed_dim % number_of_heads != 0:
            raise ValueError("Embed_dim must be divisible by NUMBER_OF_HEADS")
        self.embed_dim = embed_dim
        self.number_of_heads = number_of_heads
        self.head_dim = self.embed_dim // self.number_of_heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batchz_size = x.shape[0]
        number_of_tokens = x.shape[1]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        q = q.reshape(batchz_size, number_of_tokens, self.number_of_heads, self.head_dim) # B 256 4 16
        k = k.reshape(batchz_size, number_of_tokens, self.number_of_heads, self.head_dim)
        v = v.reshape(batchz_size, number_of_tokens, self.number_of_heads, self.head_dim)
        q = q.transpose(1, 2) # B 4 256 16
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2))
        attention = torch.softmax(scores / math.sqrt(self.head_dim), dim=-1) # B 4 256 256
        output = torch.matmul(attention, v) # B 4 256 16
        output = output.transpose(1, 2) # B 256 4 16
        output = output.reshape(batchz_size, number_of_tokens, self.embed_dim)
        output = self.output(output)

        return output, attention

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, number_of_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attention = MultiheadSelfAttention(embed_dim, number_of_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim)
        )
    
    def forward(self, x):
        attention_input = self.norm1(x)
        attention_output, attention = self.attention(attention_input)
        x = x + attention_output
        mlp_input = self.norm2(x)
        mlp_output = self.mlp(mlp_input)
        x = x + mlp_output

        return x, attention

class DualTokenViT(nn.Module):
    def __init__(self):
        super().__init__()

        patches_per_row = WIDTH // PATCH_SIZE
        patches_per_col =  HEIGHT // PATCH_SIZE
        self.number_of_patches = patches_per_row * patches_per_col

        self.patch_embedding = PatchEmbedding()

        self.pick_token = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))
        self.place_token = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))
        self.position_embedding = nn.Parameter(torch.zeros(1, self.number_of_patches + 2, EMBED_DIM))
        
        self.blocks = nn.ModuleList([TransformerBlock(EMBED_DIM, NUMBER_OF_HEADS) for _ in range(NUMBER_OF_BLOCKS)])
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.pick_head = nn.Linear(EMBED_DIM, 2)
        self.place_head = nn.Linear(EMBED_DIM, 2)

        nn.init.trunc_normal_(self.pick_token, std=0.02)
        nn.init.trunc_normal_(self.place_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, image):
        patch_tokens = self.patch_embedding(image)
        batchsize = image.shape[0]
        pick_tokens = self.pick_token.expand(batchsize, -1, -1)
        place_tokens = self.place_token.expand(batchsize, -1, -1)
        x = torch.cat([
            pick_tokens,
            place_tokens,
            patch_tokens
        ], dim=1) # B 258 64
        x = x + self.position_embedding
        last_attention = None
        for block in self.blocks:
            x, last_attention = block(x)
        x = self.norm(x)
        pick_features = x[:, 0]
        place_features = x[:, 1]
        pick_out = self.pick_head(pick_features) # B 2
        pick_out = torch.sigmoid(pick_out)
        place_out = self.place_head(place_features) # B 2
        place_out = torch.sigmoid(place_out)
        output = torch.cat([pick_out, place_out], dim=1) # B 4

        return output, last_attention

def coordinate_regression_loss(prediction, target):
    squared_error = (prediction - target) ** 2
    loss = squared_error.mean()

    return loss

def normalized_to_pixel_tensor(normalized_coordinates):
    pixel_coordinates = normalized_coordinates.clone()
    pixel_coordinates[:, 0] *= WIDTH - 1
    pixel_coordinates[:, 1] *= HEIGHT - 1
    pixel_coordinates[:, 2] *= WIDTH - 1
    pixel_coordinates[:, 3] *= HEIGHT - 1

    return pixel_coordinates

def calculate_pixel_error(prediction, target):
    prediction_pixel = normalized_to_pixel_tensor(prediction)
    target_pixel = normalized_to_pixel_tensor(target)

    pick_difference = prediction_pixel[:, 0:2] - target_pixel[:, 0:2]
    place_difference = prediction_pixel[:, 2:4] - target_pixel[:, 2:4]

    pick_distance = torch.sqrt(torch.sum(pick_difference ** 2, dim=1))
    place_distance = torch.sqrt(torch.sum(place_difference ** 2, dim=1))

    return float(pick_distance.mean().item()), float(place_distance.mean().item())


def evaluate_model(model, images, targets, device):
    model.eval()
    total_loss = 0.0
    total_pick_error = 0.0
    total_place_error = 0.0
    total_samples = 0

    with torch.no_grad():
        for start in range(0, images.shape[0], BATCH_SIZE):
            end = start + BATCH_SIZE
            batch_images = images[start:end].to(device)
            batch_targets = targets[start:end].to(device)
            prediction, _ = model(batch_images)
            loss = coordinate_regression_loss(prediction, batch_targets)
            
            batch_size = batch_images.shape[0]
            prediction_pixel = normalized_to_pixel_tensor(prediction)
            target_pixel = normalized_to_pixel_tensor(batch_targets)

            pick_difference = prediction_pixel[:, 0:2] - target_pixel[:, 0:2]
            place_difference = prediction_pixel[:, 2:4] - target_pixel[:, 2:4]

            pick_distance = torch.sqrt(torch.sum(pick_difference ** 2, dim=1))
            place_distance = torch.sqrt(torch.sum(place_difference ** 2, dim=1))

            total_loss += loss.item() * batch_size
            total_pick_error += pick_distance.sum().item()
            total_place_error += place_distance.sum().item()
            
            total_samples += batch_size

        mean_loss = total_loss / total_samples
        mean_pick_error = total_pick_error / total_samples
        mean_place_error = total_place_error / total_samples

        return mean_loss, mean_pick_error, mean_place_error


def tiny_overfit_test(train_x, train_y):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    number_of_samples = min(TINY_OVERFIT_SAMPLES, train_x.shape[0])

    tiny_x = train_x[:number_of_samples].float().to(device)
    tiny_y = train_y[:number_of_samples].float().to(device)
    
    model = DualTokenViT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)

    for step in range(TINY_OVERFIT_STEPS):
        model.train()
        prediction, _ = model(tiny_x)
        loss = coordinate_regression_loss(prediction, tiny_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_prediction, _ = model(tiny_x)
                pick_error, place_error = calculate_pixel_error(val_prediction, tiny_y)
            print("Step", step + 1, "Loss =", round(loss.item(), 6), "Pick error=", 
                  round(pick_error, 2), "Place error=", round(place_error, 2))
        
def train_model(train_x, train_y, val_x, val_y):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = DualTokenViT()
    model.to(device)

    train_x = train_x.to(device)
    train_y = train_y.to(device)
    val_x = val_x.to(device)
    val_y = val_y.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "min",
        0.5,
        4,
        0.05,
        "abs",
        1e-6
    )

    history = {
        "train_pick": [],
        "train_place": [],
        "val_pick": [],
        "val_place": []
    }
    best_validation_error = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        model.train()
        permutation = torch.randperm(train_x.shape[0])

        for start in range(0, train_x.shape[0], BATCH_SIZE):
            indices = permutation[start:start+BATCH_SIZE]
            images = train_x[indices]
            targets = train_y[indices]

            prediction, _ = model(images)
            loss = coordinate_regression_loss(prediction, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss, train_pick_error, train_place_error = evaluate_model(model, train_x, train_y, device)
        val_loss, val_pick_error, val_place_error = evaluate_model(model, val_x, val_y, device)
        validation_error = (val_pick_error + val_place_error) / 2.0
        scheduler.step(validation_error)
        current_lr = optimizer.param_groups[0]["lr"]
        improved = validation_error < best_validation_error - MIN_IMPROVEMENT
        if improved:
            best_validation_error = validation_error
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history["train_pick"].append(train_pick_error)
        history["train_place"].append(train_place_error)
        history["val_pick"].append(val_pick_error)
        history["val_place"].append(val_place_error)
        print(
            "Epoch", epoch + 1, "Train Loss =", round(train_loss, 6), 
            "Train Pick Error =", round(train_pick_error, 2),
            "Train Place Error =", round(train_place_error, 2), 
            "Val Loss =", round(val_loss, 6), 
            "Val Pick Error =", round(val_pick_error, 2),
            "Val Place Error =", round(val_place_error, 2), 
            "Lr =", current_lr
        )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("Early stopping")
            break
        
    if best_state is not None:
        model.load_state_dict(best_state)
    
    print("Best epoch:", best_epoch)
    print("Best validation error", best_validation_error)
    model = model.to("cpu")

    return model, history

def predict_pick_place_pixels(model, rgb):
    image = rgb.astype(np.float32)
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    input_tensor = torch.from_numpy(image)
    input_tensor = input_tensor.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        prediction, attention = model(input_tensor)
    
    prediction = torch.clamp(prediction, 0.0, 1.0)
    pick_u = prediction[0, 0].item() * (WIDTH - 1)
    pick_v = prediction[0, 1].item() * (HEIGHT - 1)
    place_u = prediction[0, 2].item() * (WIDTH - 1)
    place_v = prediction[0, 3].item() * (HEIGHT - 1)

    return pick_u, pick_v, place_u, place_v, attention

def pixel_to_world_on_plane(u, v, plane_z, camera_position, R_cw, K):
    pixel = np.array([
        u, 
        v, 
        1.0
    ])
    ray_camera = np.linalg.inv(K) @ pixel
    ray_world = R_cw.T @ ray_camera
    if abs(ray_world[2]) < 1e-12:
        return None
    lambda_value = (plane_z - camera_position[2]) / ray_world[2]
    if lambda_value <= 0.0:
        return None
    point_world = camera_position + ray_world * lambda_value

    return point_world

def main():

    np.random.seed(1)
    torch.manual_seed(1)

    current_folder = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_folder, "mujoco_menagerie", "universal_robots_ur5e", "ur5e_scene.xml")
    
    model, data = load_world(xml_path)

    set_robot_home(model, data)

    camera_position, R_cw, K = get_camera_geometry(model, data)

    with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
        train_x, train_y = generate_dataset(TRAIN_SAMPLES, model, data, renderer, camera_position, R_cw, K)
        val_x, val_y = generate_dataset(VAL_SAMPLES, model, data, renderer, camera_position, R_cw, K)
        show_dataset_sample(train_x[0], train_y[0])
        if RUN_TINY_OVERFIT_TEST:
            tiny_overfit_test(train_x, train_y)
        vit, history = train_model(train_x, train_y, val_x, val_y)
        model_file = os.path.join(current_folder, "dual_token_vit.pt")
        torch.save(vit.state_dict(), model_file)
        true_cube_position, true_place_position = sample_task_positions()
        set_task_positions(model, data, true_cube_position, true_place_position)
        test_rgb = capture_rgb(renderer, data)
        true_pick_pixel = world_to_pixel(true_cube_position, camera_position, R_cw, K)
        true_place_pixel = world_to_pixel(true_place_position, camera_position, R_cw, K)
        pick_u, pick_v, place_u, place_v, attention = predict_pick_place_pixels(vit, test_rgb)
        print("True Pick Pixel:", true_pick_pixel)
        print("True Place Pixel:", true_place_pixel)
        print("Predicted Pick Pixel:", (pick_u, pick_v))
        print("Predicted Place Pixel:", (place_u, place_v))
        pick_plane_z = TABLE_TOP_Z + CUBE_HALF
        place_plane_z = TABLE_TOP_Z + PLACE_HALF
        predicted_pick_position = pixel_to_world_on_plane(pick_u, pick_v, pick_plane_z, camera_position, R_cw, K)
        predicted_place_position = pixel_to_world_on_plane(place_u, place_v, place_plane_z, camera_position, R_cw, K)
        print("Predicted Pick World:", predicted_pick_position)
        print("Predicted Place World:", predicted_place_position)
        print("True Pick World", true_cube_position)
        print("True Place World", true_place_position)

if __name__ == "__main__":
    main()