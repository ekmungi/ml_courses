from sklearn.externals import joblib
from skimage import feature
import numpy as np
import cv2

# 1. Load the model
# 2. Load the image
# 3. Apply loop to perform sliding window
# 4. Extract HoG from each window
# 5. Classify using the trained model
# 6. Additionally perform step 4 & 5 on image pyramid
# 7. Store & return locations of the bounding boxes for each scale
# 8. Optionally plot the bounding boxes.

def sliding_window(image, window_size, step_size):
    
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

step_size = [1,1]
# min_wdw_sz = 

TRAIN_POSITIVES_PATH = 'D:/dev/data/vehicles/vehicles/KITTI_extracted/'
# MODEL_PATH = TRAIN_POSITIVES_PATH + 'model/linearSVC.model'
MODEL_PATH = TRAIN_POSITIVES_PATH + 'model/SVC.model'
POSITIVES_PATH = 'D:/dev/data/vehicles/positive_output/'
NEGATIVES_PATH = 'D:/dev/data/vehicles/negative_output/'

TEST_FILE = 'D:/dev/data/vehicles/test_video.mp4'

clf = joblib.load(MODEL_PATH)

cap = cv2.VideoCapture(TEST_FILE)
count_neg = 0
count_pos = 0
key = 0
while (cap.isOpened()):
    ret, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for (x, y, im_window_orig) in sliding_window(gray, [128,128], [16,16]):
        if im_window_orig.shape[0] != 128 or im_window_orig.shape[1] != 128:
            continue
        im_window = cv2.resize(im_window_orig, (64,64))
        
        H = feature.hog(im_window, orientations=9, pixels_per_cell=(10, 10), 
                        cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys", visualise=False)

        H = np.expand_dims(H, 1)
        #print('----------------------------> ', H.transpose().shape)
        pred = clf.predict(H.transpose())
        
        if pred == 1:
            file_path = POSITIVES_PATH + str(count_pos)+'.png'
            count_pos += 1
            cv2.imwrite(file_path, im_window)
            cv2.imshow('Car found', im_window_orig)
            key = cv2.waitKey(80)
        else:
            file_path = NEGATIVES_PATH + str(count_neg)+'.png'
            count_neg += 1
            cv2.imwrite(file_path, im_window)
            cv2.imshow('Car not found', im_window_orig)
            key = cv2.waitKey(1)

        if key > 0:
            break

    if key > 0:
        break
