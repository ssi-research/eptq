from datetime import datetime

FILE_TIME_STAMP = datetime.now().strftime("%d-%b-%Y__%H:%M:%S")
VAL_DIR = '/data/projects/swat/datasets_src/ImageNet/ILSVRC2012_img_val_TFrecords'
TRAIN_DIR = '/data/projects/swat/datasets_src/ImageNet/ILSVRC2012_img_train'

TRAIN_SET = 'train_set'
ALL_CLASSES = 'all_classes'
N_CLASSES = 'n_classes'
CLASSES_SUBSET = 'classes_subset'
ONE_CLASS = 'one_class'
N_IMAGES = 'n_images'
MP_PARTIAL_CANDIDATES = 'partial'
MP_FULL_CANDIDATES = 'full'

DATASET_TYPES = [TRAIN_SET, ALL_CLASSES, N_CLASSES, CLASSES_SUBSET, ONE_CLASS, N_IMAGES]

MAX_IMAGES = 1300
DEFAULT_QUANT_BITWIDTH = 8
BYTES = 4