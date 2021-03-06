# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'amazon_from_sky'))
	print(os.getcwd())
except:
	pass

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#%%
# base_path = '/media/anant/data/amazon_from_space/'
base_path = '/media/avemuri/DEV/Data/amazon_from_space/'


#%%
train_df = pd.read_csv(base_path+'train_v2.csv')
train_df.head()


#%%
# Print all unique tags
from itertools import chain
labels_list = list(chain.from_iterable([tags.split(" ") for tags in train_df['tags'].values]))
labels_set = set(labels_list)
print("There is {} unique labels including {}".format(len(labels_set), labels_set))


#%%
labels_s = pd.Series(labels_list).value_counts() # To sort them by count
fig, ax = plt.subplots(figsize=(16, 8))
sns.barplot(x=labels_s, y=labels_s.index, orient='h')


#%%
weights = labels_s.values/labels_s.values.sum()
print(weights, labels_s.index)


#%%
index = ['conventional_mine', 'road', 'habitation', 'primary', 'cloudy', 'cultivation', 'partly_cloudy', 'artisinal_mine', 'selective_logging', 'blooming', 'agriculture', 'water', 'bare_ground', 'blow_down', 'slash_burn', 'clear', 'haze']
labels_s.reindex(index)
weights = labels_s.values/labels_s.values.sum()
print(weights, labels_s.index)


#%%
print(labels_s)


#%%
reindexed = labels_s.reindex(index)


#%%
print(reindexed)
weights = 10*reindexed.values/reindexed.values.sum()
print(weights)


#%%
# for ival in index


#%%
import glob
import cv2
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, fbeta_score
from tqdm import tqdm_notebook, tqdm


import tensorflow.keras as k
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#%%
image_list = [f for f in glob.glob(base_path+'train-jpg/*.jpg')]
print(len(image_list))


#%%
X = []
y = []


df_train = pd.read_csv(base_path+'train_v2.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

for f, tags in tqdm(df_train.values, miniters=1000):
    img = cv2.imread(base_path+'train-jpg/'+'{}.jpg'.format(f))
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1 
    X.append(cv2.resize(img, (32, 32)))
    y.append(targets)
    
y = np.array(y, np.uint8)
X = np.array(X, np.float16) / 255.

print(X.shape)
print(y.shape)


#%%
SPLIT = 0.2
dataset_size = len(X)
indices = list(range(dataset_size))
split = int(np.floor(SPLIT * dataset_size))

np.random.seed(42)
np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]
x_train, x_valid, y_train, y_valid = X[train_indices], X[val_indices], y[train_indices], y[val_indices]
print(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)


#%%
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(32, 32, 3)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(17, activation='sigmoid'))


#%%
class KerasMetrics:
	def precision(self, y_true, y_pred):
		true_positives = k.backend.sum(k.backend.round(k.backend.clip(y_true * y_pred, 0, 1)))
		predicted_positives = k.backend.sum(k.backend.round(k.backend.clip(y_pred, 0, 1)))
		precision = true_positives / (predicted_positives + k.backend.epsilon())
		return precision

	def recall(self, y_true, y_pred):
		true_positives = k.backend.sum(k.backend.round(k.backend.clip(y_true * y_pred, 0, 1)))
		possible_positives = k.backend.sum(k.backend.round(k.backend.clip(y_true, 0, 1)))
		recall = true_positives / (possible_positives + k.backend.epsilon())
		return recall

	def fbeta_score(self, y_true, y_pred, beta=2):
		if beta < 0:
			raise ValueError('The lowest choosable beta is zero (only precision).')

		if k.backend.sum(k.backend.round(k.backend.clip(y_true, 0, 1))) == 0:
			return 0

		p = self.precision(y_true, y_pred)
		r = self.recall(y_true, y_pred)
		bb = beta ** 2
		fbeta_score = (1 + bb) * (p * r) / (bb * p + r + k.backend.epsilon())
		return fbeta_score


#%%
keras_metrics = KerasMetrics()
# We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
optim = k.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy', keras_metrics.fbeta_score])


#%%
datagen = k.preprocessing.image.ImageDataGenerator(width_shift_range=0.2,
                                                   height_shift_range=0.2,
                                                   vertical_flip=True)

datagen.fit(x_train)


#%%
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=256),
                              steps_per_epoch=len(x_train) / 256, epochs=50,
                              validation_data=(x_valid, y_valid))


#%%
# history = model.fit(x_train, y_train,
#                     batch_size=256,
#                     epochs=50,
#                     verbose=1,
#                     validation_data=(x_valid, y_valid))


#%%
_ = plt.plot(history.history['loss'], label='Train')
_ = plt.plot(history.history['val_loss'], label='Validation')
_ = plt.title('Loss')
_ = plt.legend()


#%%
_ = plt.plot(history.history['fbeta_score'], label='Train')
_ = plt.plot(history.history['val_fbeta_score'], label='Validation')
_ = plt.title('fbeta score')
_ = plt.legend()

#%%
_ = plt.plot(history.history['acc'], label='Train')
_ = plt.plot(history.history['val_acc'], label='Validation')
_ = plt.title('Accuracy')
_ = plt.legend()


#%%
from tensorflow.keras import applications

#%%
img_width, img_height = 256, 256
model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

#%%
model.summary()

#%%
for layer in model.layers[:5]:
    layer.trainable = False

#%%
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(17, activation="sigmoid")(x)

#%%
model_final = Model(inputs=model.input, outputs=predictions)

#%%

keras_metrics = KerasMetrics()
# We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
optim = k.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy', keras_metrics.fbeta_score])


#%%

history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                              steps_per_epoch=len(x_train) / 32, epochs=50,
                              validation_data=(x_valid, y_valid))











#%%
keras_metrics = KerasMetrics()
# We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
optim = k.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model_final.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy', keras_metrics.fbeta_score])






class AmazonDataset(K.utils.Sequence):
    def __init__(self, base_folder, csv_file, folder_name, transform=None):
        self.csv_file = csv_file
        self.labels_df = pd.read_csv(base_path+csv_file)
        self.base_folder = base_folder
        self.image_list = [f for f in glob.glob(base_path+folder_name+'/*.jpg')]
        self.transform = transform
        self.mlb = MultiLabelBinarizer()
        self.labels = self.mlb.fit_transform(self.labels_df['tags'].str.split()).astype(np.float32)
        
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):

        batch_paths = np.random.choice(a = np.arange(len(image_list)), 
                                        size = batch_size)
        
        batch_input = []
        batch_output = [] 
        
        # Read in each input, perform preprocessing and get labels
        for index in batch_paths:
            image = cv2.imread(image_list[index])
            # input = preprocess_input(image=input)
            batch_input += [ image ]
            batch_output += [ labels[index] ]
        # Return a tuple of (input,output) to feed the network
        batch_x = np.array( batch_input )
        batch_y = np.array( batch_output )
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label


#%%
# train_df = pd.read_csv(base_path+'train_v2.csv')
# datagen=ImageDataGenerator(rescale=1./255)
# train_datagen = datagen.flow_from_dataframe(dataframe=train_df, directory=base_path+'train-jpg', 
#                                             x_col='image_name', y_col='tags', class_mode="categorical",
#                                             target_size=(256,256), batch_size=32)


#%%
# for i in range(4):
#     x, y = next(train_datagen)



#%%
image_list = [f for f in glob.glob(base_path+'train-jpg/*.jpg')]
labels_df = pd.read_csv(base_path+'train_v2.csv')
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels_df['tags'].str.split()).astype(np.float32)

SPLIT = 0.2
dataset_size = len(image_list)
indices = list(range(dataset_size))
split = int(np.floor(SPLIT * dataset_size))

np.random.seed(42)
np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]



#%%
def image_generator(image_list, labels, batch_size=64, selection_indices=None):
    
    while True:
        # Select files (paths/indices) for the batch
        if selection_indices is None:
            batch_paths = np.random.choice(a=np.arange(len(image_list)), 
                                            size=batch_size)
        else:
            batch_paths = np.random.choice(a=selection_indices, 
                                            size=batch_size)

        batch_input = []
        batch_output = [] 
        
        # Read in each input, perform preprocessing and get labels
        for index in batch_paths:
            image = cv2.resize(cv2.imread(image_list[index]), (256, 256))
            # input = preprocess_input(image=input)
            batch_input += [ image ]
            batch_output += [ labels[index] ]
        # Return a tuple of (input,output) to feed the network
        batch_x = np.array( batch_input )
        batch_y = np.array( batch_output )
        
        yield( batch_x, batch_y )

#%%
for i in range(4):
    X, y = next(image_generator(image_list, labels, batch_size=16, selection_indices=train_indices))
    print("i: ", X.shape, y.shape)
#%%
keras_metrics = KerasMetrics()
# We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
optim = k.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model_final.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy', keras_metrics.fbeta_score])


#%%
history = model_final.fit_generator(image_generator(image_list, labels, batch_size=16, selection_indices=train_indices),
                                    steps_per_epoch=len(train_indices)/16, epochs=10,
                                    validation_data=image_generator(image_list, labels, batch_size=16, selection_indices=val_indices),
                                    validation_steps=len(val_indices)/16)


#%%
# model_final.summary()
print(len(model_final.layers[-3:]))


# for layer in model_final.layers[:-3]:
#     layer.trainable = False
#%%


#%%



#%%




#%%
datagen = k.preprocessing.image.ImageDataGenerator(width_shift_range=0.2,
                                                   height_shift_range=0.2,
                                                   vertical_flip=True)

it = datagen.flow(image_generator(base_path, 'train_v2.csv', 'train-jpg', batch_size=1), batch_size=16)
#%%

#%%

#%%

#%%


#%%
#%%
gen = DataGenerator()


#%%
def csv_image_generator(inputPath, bs, lb, mode="train", aug=None):
	# open the CSV file for reading
	f = open(inputPath, "r")

	# loop indefinitely
	while True:
		# initialize our batches of images and labels
		images = []
		labels = []

		# keep looping until we reach our batch size
		while len(images) < bs:
			# attempt to read the next line of the CSV file
			line = f.readline()

			# check to see if the line is empty, indicating we have
			# reached the end of the file
			if line == "":
				# reset the file pointer to the beginning of the file
				# and re-read the line
				f.seek(0)
				line = f.readline()

				# if we are evaluating we should now break from our
				# loop to ensure we don't continue to fill up the
				# batch from samples at the beginning of the file
				if mode == "eval":
					break

			# extract the label and construct the image
			line = line.strip().split(",")
			label = line[0]
			image = np.array([int(x) for x in line[1:]], dtype="uint8")
			image = image.reshape((64, 64, 3))

			# update our corresponding batches lists
			images.append(image)
			labels.append(label)

		# one-hot encode the labels
		labels = lb.transform(np.array(labels))

		# if the data augmentation object is not None, apply it
		if aug is not None:
			(images, labels) = next(aug.flow(np.array(images),
				labels, batch_size=bs))

		# yield the batch to the calling function
		yield (np.array(images), labels)


