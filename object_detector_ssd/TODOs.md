# ToDos for Object detection in images.

1. [ ] Using pytorch to create a single object classification model.
   1. [x] MNIST
   2. [x] FashionMNIST
   3. [x] Fruits-360 images from Kaggle
2. [ ] Extend the single object classification to multiple objects/label classification in an image.
   1. [ ] Explore this in the following dataset:
      - <https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/>
   2. [] 
3. Using the pytorch SSD model from https://github.com/qfgaohao/pytorch-ssd. 
    1. [ ] Detect single objects in a scene
    2. [ ] Retrain for blood cell detection
    3. [ ] How can it be extended to instance detection.
    4. [ ] Use instance detection in video images.
    5. [ ] Explore TensorFlow object detection API
        - <https://gilberttanner.com/category/videos/tensorflow-object-detection/>
4. Faces
    1. Face detection
        - <http://tamaraberg.com/faceDataset/index.html>
        - <https://www.kaggle.com/ciplab/real-and-fake-face-detection>
        - <https://medium.com/diving-in-deep/facial-keypoints-detection-with-pytorch-86bac79141e4>
    2. Face recognition
        - Siamese network: 
        - <https://medium.com/hackernoon/one-shot-learning-with-siamese-networks-in-pytorch-8ddaab10340e?source=search_post---------3>
        - <https://medium.com/hackernoon/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7?source=search_post---------6>
5. Combine 1 & 2 to do in video person detection and then person identification.
6. Port this to Jetson Nano.
7. Home surveillance
    1. TensorFlow
    - Check <https://gilberttanner.com/2019/07/20/simple-surveillance-system-with-the-tensorflow-object-detection-api/>
    - <https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#2-set-up-tensorflow-directory-and-anaconda-virtual-environment>
