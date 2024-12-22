import torch
from torch.utils.data import default_collate
import torch.nn.functional as F


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    # first pass to gather statistics
    max_length_audio = 0
    max_length_spectrogram = 0
    max_length_encoded_text = 0

    for dataset_item in dataset_items:
        # (1, audio_length)
        max_length_audio = max(max_length_audio, dataset_item["audio"].size(1))
        # (1, samples, length)
        max_length_spectrogram = max(
            max_length_spectrogram, dataset_item["spectrogram"].size(2)
        )
        # (1, encoded_text_length)
        max_length_encoded_text = max(
            max_length_encoded_text, dataset_item["text_encoded"].size(1)
        )

    # pad each of the tensors to the max length and also save original length
    for dataset_item in dataset_items:
        dataset_item["text_encoded_length"] = dataset_item["text_encoded"].size(1)
        dataset_item["spectrogram_length"] = dataset_item["spectrogram"].size(2)

        dataset_item["audio"] = F.pad(
            dataset_item["audio"],
            (0, max_length_audio - dataset_item["audio"].size(1)),
            value=0,
        ).squeeze(0)
        dataset_item["spectrogram"] = F.pad(
            dataset_item["spectrogram"],
            (0, max_length_spectrogram - dataset_item["spectrogram"].size(2)),
            value=0,
        ).squeeze(0)
        dataset_item["text_encoded"] = F.pad(
            dataset_item["text_encoded"],
            (0, max_length_encoded_text - dataset_item["text_encoded"].size(1)),
            value=0,
        ).squeeze(0)

    return default_collate(dataset_items)
