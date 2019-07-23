import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.externals import joblib
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


MODEL_PATH = TRAIN_POSITIVES_PATH + 'model/'


train_positives = natsorted(glob(TRAIN_POSITIVES_FEATURES_PATH+'*.feat'), alg=ns.IGNORECASE)
train_negatives = natsorted(glob(TRAIN_NEGATIVES_FEATURES_PATH+'*.feat'), alg=ns.IGNORECASE)
train_negatives_extra = natsorted(glob(TRAIN_NEGATIVES_FEATURES_PATH_EXTRA+'*.feat'), alg=ns.IGNORECASE)
false_positives = natsorted(glob(FALSE_POSITIVES_FEATURES_PATH+'*.feat'), alg=ns.IGNORECASE)

train_desc = []
train_label = []
print('\n\n Loading positive features...')
for feat_path in tqdm(train_positives):
    desc = np.load(feat_path)
    train_desc.append(desc)
    train_label.append([1])

print('\n\n Loading negative features...')
for feat_path in tqdm(train_negatives):
    desc = np.load(feat_path)
    train_desc.append(desc)
    train_label.append([0])

print('\n\n Loading extra negative features...')
for feat_path in tqdm(train_negatives_extra):
    desc = np.load(feat_path)
    train_desc.append(desc)
    train_label.append([0])

print('\n\n Loading false positive features...')
for feat_path in tqdm(false_positives):
    desc = np.load(feat_path)
    train_desc.append(desc)
    train_label.append([0])

train_label = np.squeeze(np.array(train_label))
train_desc = np.array(train_desc)
print(train_desc.shape)
print('{0} descriptors loaded each of size {1}'.format(train_desc.shape[0], train_desc.shape[1]))

print('{0} positive features, {1} negative features loaded.'.format(len(train_positives), 
                        len(train_negatives)+len(false_positives)+len(train_negatives_extra)))

print('Training...')
# clf = LinearSVC(random_state=42)
clf = SVC(random_state=42)
clf.fit(train_desc, train_label)
print('Done.')

if not os.path.isdir(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# joblib.dump(clf, MODEL_PATH+'linearSVC.model')
# print('Saving file to: {}'.format(MODEL_PATH+'linearSVC.model'))

joblib.dump(clf, MODEL_PATH+'SVC.model')
print('Saving file to: {}'.format(MODEL_PATH+'SVC.model'))