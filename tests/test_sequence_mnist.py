import torch
from transformers import TrOCRProcessor

from sequence_mnist.model import SequenceMNIST

NUM_DIGITS = 5  # do not change

def test_sample(processor = None):
    """
    Tests types and dimensionality of sample
    """
    if not processor:
        processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-printed')
    under_test = SequenceMNIST(train=True,  processor=processor, download=True)

    sample = under_test[0]

    # DIMENSIONALITY TESTS
    assert sample['pixel_values'].shape == (3,384,384)
    assert len(sample['labels']) == 5
    assert len(sample['text']) == 5

    # TYPE TESTS
    assert type(sample['pixel_values']) == torch.Tensor
    assert type(sample['labels']) == torch.Tensor
    assert type(sample['text']) == str