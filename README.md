# 6D pose estimation project

**Model Checkpoints** available at this [link](https://drive.google.com/file/d/1OKD_XCv4D7nL7pG6inpUZaMsEdSe1LJ_/view?usp=sharing).
**Yolo Checkpoints** are already included in the repository.

The approach we use is a keypoint-based approach. We basically sample the 3D model mesh and project the sampled points onto the 2D image plane. The model is trained to predict the 2D coordinates of these keypoints (using only the cropped rgb image), which are then used to estimate the 6D pose of the object using the PnP algorithm. The loss function that we use

Our extension over the original model is a pose-refinement module that takes the predicted s and the cropped rgb image as input and predicts the 3D pose of the object. The pose-refinement module is trained to minimize the ADD metric, which is a common metric for evaluating 6D pose estimation models.

Notebooks for testing the project are the following:

- *yolo.ipynb*: training and testing the Yolo model if you plan to not use the pre-trained checkpoints you need to run this notebook before proceeding to run our model.
- *training.ipynb*: runs the training of our model (the available checkpoints are trained over ~150 epochs)
- *testing.ipynb*: testing the model both visually and by the means of ADD metric it also uses our pose-refinement module which is our extension over the initial model.

### TODOs

- ADD-S metric for symmetric objects (should be easy to implement)
- Code refactoring
- training and testing the model different training-testing splits
- There are some issues in the conversion from the original dataset to the yolo format, so some data from the original dataset is missing when training our pose prediction model.
