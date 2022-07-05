import numpy as np
from sklearn.utils import shuffle

from PIL import Image

import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers.models.trocr.processing_trocr import TrOCRProcessor
from sequence_mnist.data.mnist import MNIST

NUM_DIGITS = 5  # do not change


class SequenceMNIST(MNIST, Dataset):
    """MNIST Sequence Dataset Class"""

    def __init__(self, *, train: bool, processor: TrOCRProcessor, root: str = "/tmp/data",
                download: bool = False, max_target_length: int = 5):
        super(SequenceMNIST, self).__init__(train=train, root=root, download=download)
        self.num_digits = NUM_DIGITS
        self.processor = processor
        self.max_target_length = max_target_length
        self.shuffle_data()
        self.transforms = transforms.RandomApply([
                                                  transforms.RandomRotation(degrees=15),
                                                  transforms.GaussianBlur(kernel_size=3)
                                                  ], p=0.5)


    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Transform and augment an image
        Args:
            image : MNIST Image
        Returns:
            image : MNIST Image with randomly applied transformations
        """
        return self.transforms(torch.Tensor(image).unsqueeze(dim=0))

    def preprocess_seq(self, images: torch.Tensor) -> torch.Tensor:
        """Process image sequence into pixel value encoding for input to TrOCR
        Args:
            images : Concatenated sequence of images
        Returns:
            pixel_values: Encoded sequence of images
        """
        pixel_values = self.processor(images=images.repeat(3, 1, 1), return_tensors="pt").pixel_values
        return pixel_values.squeeze()

    def process_ids(self, generated_ids) -> np.array:
        """Transform outputed IDs to human readable text"""
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text

    def shuffle_data(self) -> None:
        """Shuffle data and targets randomly"""
        self.data, self.targets = shuffle(self.data, self.targets)

    def display_image(self, idx):
        plt.imshow(self.data[idx], cmap="gray")

    def display_sequence(self, images: torch.Tensor):
        plt.imshow(images.permute(1,2,0), cmap='gray')

    def __getitem__(self, idx) -> "tuple[np.ndarray, np.ndarray]":
        """Get pixel values of image sequence, with label IDs and human readable text"""
        
        start = idx * self.num_digits
        end = start + self.num_digits

        # Select images
        images = [self.preprocess(img) for img in self.data[start:end]]
        # Stack
        images = torch.cat(images, dim=2)
        assert images.shape == (1, 28, 28 * self.num_digits)
        # Process for input to TrOCR
        pixel_values = self.preprocess_seq(images)
        # Get labels
        text = "".join([str(label) for label in self.targets[start:end]])
        # Tokenise text labels
        labels = self.processor.tokenizer(text, 
                                          padding="max_length", 
                                          max_length=self.max_target_length).input_ids
        # PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        return {'pixel_values' : pixel_values,
                'labels' : torch.LongTensor(labels),
                'text' : text}
    
    def __len__(self) -> int:
        return len(self.data) // self.num_digits
