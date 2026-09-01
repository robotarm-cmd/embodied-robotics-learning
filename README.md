# embodied-robotics-learning
My Python learning code for robotic arms, PyBullet and embodied AI.
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
"""
<img width="1500" height="600" alt="image" src="https://github.com/user-attachments/assets/135cd9e4-5f93-4e1e-a102-e98e83f60083" />

In the version of “train_cnn_v1.py”, we have incorporated the scenario where the robotic arm occlude the cube during training. By changing the posture of the robotic arm, we supplement the images captured by the camera. Through the use of the blocking filter, we generate training set images. We also changed the calculation method of u and v in the first version. Even if it is blocked, we can still provide accurate u and v calculations in the training set, further enhancing the effect in the test set.
