import os
from typing import List, Callable

import numpy as np
from PIL import Image

#:
FILETYPES = ['jpeg', 'jpg', 'bmp', 'png']


class FolderImageLoader(object):
    """

    Class for images loading, processing and retrieving.

    """

    def __init__(self,
                 folder: str,
                 preprocessing: List[Callable],
                 batch_size: int,
                 random_batch: bool = True,
                 file_types: List[str] = FILETYPES):


        self.folder = folder
        self.image_list = []
        self.random_batch = random_batch
        self.index = 0
        # print(f"Starting Scanning Disk: {self.folder}")
        for root, dirs, files in os.walk(self.folder):
            for file in files:
                file_type = file.split('.')[-1].lower()
                if file_type in file_types:
                    self.image_list.append(os.path.join(root, file))
        self.n_files = len(self.image_list)
        assert self.n_files > 0, f'Folder to load can not be empty.'
        # print(f"Finished Disk Scanning: Found {self.n_files} files")
        self.preprocessing = preprocessing
        self.batch_size = batch_size

    def _sample(self):
        """
        Read batch_size random images from the image_list the FolderImageLoader holds.
        Process them using the preprocessing list that was passed at initialization, and
        prepare it for retrieving.
        """

        if self.random_batch:
            index = np.random.randint(0, self.n_files, self.batch_size)
        else:
            index = self.index + np.arange(0, self.batch_size)
            self.index += self.batch_size
            if self.index >= self.n_files:
                self.index -= self.n_files

        image_list = []
        for i in index:
            file = self.image_list[i]
            img = np.uint8(np.array(Image.open(file).convert('RGB')))
            for p in self.preprocessing:  # preprocess images
                img = p(img)
            image_list.append(img)
        self.next_batch_data = np.stack(image_list, axis=0)

    def reset(self, index=0):
        self.index = index

    def sample(self):
        """

        Returns: A sample of batch_size images from the folder the FolderImageLoader scanned.

        """
        self._sample()
        data = self.next_batch_data  # get current data
        return data
