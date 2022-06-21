from datetime import datetime

FILE_TIME_STAMP = datetime.now().strftime("%d-%b-%Y__%H:%M:%S")
VAL_DIR = '/data/projects/swat/datasets_src/ImageNet/ILSVRC2012_img_val_TFrecords'
TRAIN_DIR = '/data/projects/swat/datasets_src/ImageNet/ILSVRC2012_img_train'

ALL_CLASSES = 'all_classes'
N_CLASSES = 'n_classes'
CLASSES_SUBSET = 'classes_subset'
ONE_CLASS = 'one_class'
N_IMAGES = 'n_images'

DATASET_TYPES = [ALL_CLASSES, N_CLASSES, CLASSES_SUBSET, ONE_CLASS, N_IMAGES]

MAX_IMAGES = 1300