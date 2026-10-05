# embodied-robotics-learning
My Python learning code for robotic arms, PyBullet, and embodied AI randomly generates training images in PyBullet, automatically obtains the ground-truth pixel location of the cube, creates the corresponding heatmap labels, feeds the RGB images into a CNN to predict heatmaps, calculates the loss between the predicted and ground-truth heatmaps, performs backpropagation, updates the CNN parameters.


In the version of “train_cnn_v1.py”, we have incorporated the scenario where the robotic arm occlude the cube during training. By changing the posture of the robotic arm, we supplement the images captured by the camera. Through the use of the blocking filter, we generate training set images. We also changed the calculation method of u and v in the first version. Even if it is blocked, we can still provide accurate u and v calculations in the training set, further enhancing the effect in the test set.

Then, import the model of “train_cnn_v1.py” and conduct validation in “cnn_ur5.py” on the test set.

"pick_place_unet_v1.py" mainly underwent the following modifications: the place placement position was added for training, so the heatmap became two images. It was difficult to train using only CNN because it can only recognize local features. This time, we sampled the Unet network for training and achieved good results.

"action_map_to_ur5.py" mainly use results obtained by "pick_place_unet_v1.py" to realize pick and place for a cube.

We migrated the PyBullet to MuJoCo. 

"vit_place_pick_v1.py" uses a SingleHeadSelfAttention mechnism to train the pick and place position of the world, meanwhile, the one cls_token is used to predict pick and place position together. However, the effectiveness of the train_model is not good, it will enter a plateau period along with time. Then, we will check the model problem next time.

"vit_place_pick_v2.py" introduces separate Pick and Place tokens, allowing each task to learn its own task-specific representation. V2 also replaces single-head attention with multi-head attention, increases the Transformer depth, and uses a more complete training strategy with AdamW, weight decay, learning-rate scheduling, early stopping, and a larger dataset. The main advantage of V2 is better task separation and representation capacity. The Pick token can focus more on features related to the object to grasp, while the Place token can focus on the target region. Multi-head attention further allows the model to capture different spatial relationships simultaneously, while the improved training pipeline gives more stable and reliable convergence.
