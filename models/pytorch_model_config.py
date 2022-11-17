from collections import Callable
import timm
import torch
import numpy as np
from datasets.image_utils import pytorch_model_accuracy_evaluation
from torch.utils.data import Subset


class ModelParameters(object):
    def __init__(self,
                 model,
                 float_accuracy,
                 preprocess=None,
                 evaluation_function=pytorch_model_accuracy_evaluation,
                 model_params={},
                 interpolation="bilinear",
                 image_size=(3, 224, 224),
                 name=None,
                 ):
        self.name = name
        self.model = model
        self.float_accuracy = float_accuracy
        self.preprocess = preprocess
        self.evaluation_function = evaluation_function
        self.model_params = model_params
        self.interpolation = interpolation
        self.image_size = image_size
        self.resize_to = [self.image_size[1], self.image_size[2]]

    def get_float_accuracy(self):
        return self.float_accuracy

    def get_model(self):
        return self.model(**self.model_params)

    def get_validation_dataset(self, dir_path, batch_size=32, image_size=(224, 224)):
        ds = timm.data.create_dataset("", dir_path)
        return timm.data.create_loader(ds,
                                       input_size=(3, *image_size),
                                       batch_size=batch_size,
                                       is_training=False,
                                       use_prefetcher=False,
                                       interpolation=self.interpolation,
                                       tf_preprocessing=False)

    def get_representative_dataset_from_classes_and_images(self, in_dir, num_images,
                                                           image_size=224, batch_size=1,
                                                           augmentation_pipepline: Callable = None):

        transform = timm.data.create_transform(image_size, interpolation=self.interpolation, color_jitter=None,
                                               is_training=True)
        ds = timm.data.create_dataset("", in_dir, transform=transform)
        ds = Subset(ds, list(np.random.randint(0, len(ds) + 1, num_images)))
        dl = torch.utils.data.DataLoader(ds, batch_size, shuffle=True, num_workers=4)

        class RepresentativeDataset(object):
            def __init__(self, in_data_loader):
                self.dl = in_data_loader
                self.iter = iter(self.dl)

            def __call__(self):
                try:
                    x = next(self.iter)[0]
                except StopIteration:
                    self.iter = iter(self.dl)
                    x = next(self.iter)[0]
                if augmentation_pipepline is not None:
                    x = augmentation_pipepline(x)
                return [x.cpu().numpy()]

        return RepresentativeDataset(dl)

    def get_representative_dataset(self, representative_dataset_folder, batch_size, n_images, image_size,
                                   preprocessing=None, seed=0, debug: bool = False,
                                   augmentation_pipepline: Callable = None):
        if debug:
            x = np.random.randn(batch_size, 3, image_size, image_size)

            def representative_dataset():
                return [x]

            return representative_dataset

        return self.get_representative_dataset_from_classes_and_images(
            in_dir=representative_dataset_folder,
            num_images=n_images,
            batch_size=batch_size,
            augmentation_pipepline=augmentation_pipepline)
