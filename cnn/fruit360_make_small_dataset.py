import os

ROOT_PATH = 'D:/dev/data/fruits-360/'

def mkdir(p):
    if not os.path.exists(p):
        os.mkdir(p)


def link(src, dest):
    if not os.path.exists(dest):
        os.symlink(src, dest, target_is_directory=True)

# mkdir(ROOT_PATH+'small-set/')

classes = ['Apple Red 1','Banana','Avacado']

train_path_from = os.path.abspath(ROOT_PATH+'large_set/Training/')
valid_path_from = os.path.abspath(ROOT_PATH+'large_set/Validation/')

train_path_to = os.path.abspath(ROOT_PATH+'small_set/Training/')
valid_path_to = os.path.abspath(ROOT_PATH+'small_set/Validation/')


mkdir(train_path_to)
mkdir(valid_path_to)

for c in classes:
    link(train_path_from+'/'+c, train_path_to+'/'+c)
    link(valid_path_from+'/'+c, valid_path_to+'/'+c)


