# embodied-robotics-learning
My Python learning code for robotic arms, PyBullet, and embodied AI randomly generates training images in PyBullet, automatically obtains the ground-truth pixel location of the cube, creates the corresponding heatmap labels, feeds the RGB images into a CNN to predict heatmaps, calculates the loss between the predicted and ground-truth heatmaps, performs backpropagation, updates the CNN parameters.

"""
<img width="1500" height="600" alt="image" src="https://github.com/user-attachments/assets/135cd9e4-5f93-4e1e-a102-e98e83f60083" />

In the version of “train_cnn_v1.py”, we have incorporated the scenario where the robotic arm occlude the cube during training. By changing the posture of the robotic arm, we supplement the images captured by the camera. Through the use of the blocking filter, we generate training set images. We also changed the calculation method of u and v in the first version. Even if it is blocked, we can still provide accurate u and v calculations in the training set, further enhancing the effect in the test set.
<img width="750" height="615" alt="3789d512-82b9-429d-9fe0-2a8fdbceefb3" src="https://github.com/user-attachments/assets/a73bc514-1690-4300-9978-773497f1de19" />

Then, import the model of “train_cnn_v1.py” and conduct validation in “cnn_ur5.py” on the test set.
