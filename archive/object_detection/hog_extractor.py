import numpy as np
import cv2
import sklearn
import matplotlib.pyplot as plt
from skimage import feature
from skimage import exposure
import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted, ns

# NOTE: All images are 64x64 in this database
TRAIN_POSITIVES_PATH = 'D:/dev/data/vehicles/vehicles/KITTI_extracted/'
TRAIN_NEGATIVES_PATH = 'D:/dev/data/vehicles/non-vehicles/GTI/'
TRAIN_NEGATIVES_PATH_EXTRA = 'D:/dev/data/vehicles/non-vehicles/Extras/'
FALSE_POSITIVES_PATH = 'D:/dev/data/vehicles/non-vehicles/false_positives/'

TRAIN_POSITIVES_FEATURES_PATH = TRAIN_POSITIVES_PATH + 'features/'
TRAIN_NEGATIVES_FEATURES_PATH = TRAIN_NEGATIVES_PATH + 'features/'
TRAIN_NEGATIVES_FEATURES_PATH_EXTRA = TRAIN_NEGATIVES_PATH_EXTRA + 'features/'
FALSE_POSITIVES_FEATURES_PATH = FALSE_POSITIVES_PATH + 'features/'


if not os.path.exists(TRAIN_POSITIVES_FEATURES_PATH):
    os.makedirs(TRAIN_POSITIVES_FEATURES_PATH)
if not os.path.exists(TRAIN_NEGATIVES_FEATURES_PATH):
    os.makedirs(TRAIN_NEGATIVES_FEATURES_PATH)
if not os.path.exists(TRAIN_NEGATIVES_FEATURES_PATH_EXTRA):
    os.makedirs(TRAIN_NEGATIVES_FEATURES_PATH_EXTRA)
if not os.path.exists(FALSE_POSITIVES_FEATURES_PATH):
    os.makedirs(FALSE_POSITIVES_FEATURES_PATH)

train_positives = natsorted(glob(TRAIN_POSITIVES_PATH+'*.png'), alg=ns.IGNORECASE)
train_negatives = natsorted(glob(TRAIN_NEGATIVES_PATH+'*.png'), alg=ns.IGNORECASE)
train_negatives_extra = natsorted(glob(TRAIN_NEGATIVES_PATH_EXTRA+'*.png'), alg=ns.IGNORECASE)
false_positives = natsorted(glob(FALSE_POSITIVES_PATH+'*.png'), alg=ns.IGNORECASE)

print('Extracting features from positive samples...')
for image_path in tqdm(train_positives):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (H, hogImage_L) = feature.hog(gray, orientations=9, pixels_per_cell=(10, 10), 
                                cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys", visualise=True)

    feat_loc = TRAIN_POSITIVES_FEATURES_PATH + os.path.split(image_path)[1].split('.')[0] + '.feat'
    H.dump(feat_loc)

print('Extracting features from negative samples...')
for image_path in tqdm(train_negatives):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (H, hogImage_L) = feature.hog(gray, orientations=9, pixels_per_cell=(10, 10), 
                                cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys", visualise=True)

    feat_loc = TRAIN_NEGATIVES_FEATURES_PATH + os.path.split(image_path)[1].split('.')[0] + '.feat'
    H.dump(feat_loc)

print('Extracting features from extra negative samples...')
for image_path in tqdm(train_negatives_extra):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (H, hogImage_L) = feature.hog(gray, orientations=9, pixels_per_cell=(10, 10), 
                                cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys", visualise=True)

    feat_loc = TRAIN_NEGATIVES_FEATURES_PATH_EXTRA + os.path.split(image_path)[1].split('.')[0] + '.feat'
    H.dump(feat_loc)

print('Extracting features from false positive samples...')
for image_path in tqdm(false_positives):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (H, hogImage_L) = feature.hog(gray, orientations=9, pixels_per_cell=(10, 10), 
                                cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys", visualise=True)
    feat_loc = FALSE_POSITIVES_FEATURES_PATH + os.path.split(image_path)[1].split('.')[0] + '.feat'
    H.dump(feat_loc)


# imagePath = 'hockey_players.jpg'

# image = cv2.imread(imagePath)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

# # extract Histogram of Oriented Gradients from the test image and
# # predict the make of the car
# (H, hogImage_L) = feature.hog(gray[:,:,0], orientations=9, pixels_per_cell=(10, 10),
#     cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualise=True)
# (H, hogImage_U) = feature.hog(gray[:,:,1], orientations=9, pixels_per_cell=(10, 10),
#     cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualise=True)
# (H, hogImage_V) = feature.hog(gray[:,:,2], orientations=9, pixels_per_cell=(10, 10),
#     cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualise=True)
# # pred = model.predict(H.reshape(1, -1))[0]

# # visualize the HOG image
# hogImage_L = exposure.rescale_intensity(hogImage_L, out_range=(0, 255))
# hogImage_U = exposure.rescale_intensity(hogImage_U, out_range=(0, 255))
# hogImage_V = exposure.rescale_intensity(hogImage_V, out_range=(0, 255))
# hogImage_L = hogImage_L.astype("uint8")
# hogImage_U = hogImage_U.astype("uint8")
# hogImage_V = hogImage_V.astype("uint8")
# cv2.imshow("HOG image L", hogImage_L)
# cv2.imshow("HOG image U", hogImage_U)
# cv2.imshow("HOG image V", hogImage_V)

# # draw the prediction on the test image and display it
# # cv2.putText(image, pred.title(), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
#     # (0, 255, 0), 3)
# cv2.imshow("Original image", image)
# cv2.waitKey(0)