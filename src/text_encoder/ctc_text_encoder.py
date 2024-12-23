import re
from string import ascii_lowercase
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
import os
import torch
import pandas as pd
import numpy as np


# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class BPETextEncoder:
    def __init__(
        self,
        data_file_path: str,
        vocab_file_path: str,
        sp_model_prefix: str = None,
        vocab_size: int = 2000,
        normalization_rule_name: str = "nmt_nfkc_cf",
        model_type: str = "bpe",
        pad_id=0,
    ):
        """
        Dataset with texts, supporting BPE tokenizer
        :param data_file: json of librispeech
        :param train: whether to use train or validation split
        :param sp_model_prefix: path prefix to save tokenizer model
        :param vocab_size: sentencepiece tokenizer vocabulary size
        :param normalization_rule_name: sentencepiece tokenizer normalization rule
        :param model_type: sentencepiece tokenizer model type
        :param max_length: maximal length of text in tokens
        """
        if not os.path.isfile(sp_model_prefix + ".model"):
            if not os.path.isfile(vocab_file_path):
                data = pd.read_json(data_file_path)
                with open(vocab_file_path, "w") as file:
                    file.write("\n".join(data["text"].to_list()))

            # train tokenizer if not trained yet
            SentencePieceTrainer.train(
                input=vocab_file_path,
                vocab_size=vocab_size,
                model_type=model_type,
                model_prefix=sp_model_prefix,
                normalization_rule_name=normalization_rule_name,
                pad_id=pad_id,
            )
        # load tokenizer from file
        self.sp_model = SentencePieceProcessor(model_file=sp_model_prefix + ".model")

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        return torch.Tensor(self.sp_model.encode(text)).long().unsqueeze(0)

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        if isinstance(inds, np.ndarray):
            assert (
                len(inds.shape) <= 2
            ), "Expected tensor of shape (length, ) or (batch_size, length)"
            inds = inds.tolist()
        return self.sp_model.decode(inds)

    def ctc_decode(self, inds) -> str:
        if isinstance(inds, np.ndarray):
            assert (
                len(inds.shape) <= 2
            ), "Expected tensor of shape (length, ) or (batch_size, length)"
            inds = inds.tolist()

        decoded = []
        previous_token = self.sp_model.pad_id()
        for ind in inds:
            if ind != previous_token and ind != self.sp_model.pad_id():
                decoded.append(self.sp_model.IdToPiece(ind))
            previous_token = ind
        print(decoded)
        return "".join(decoded).replace("▁", " ").strip()

    def get_pad_id(self):
        return self.sp_model.pad_id()

    def __len__(self):
        return self.sp_model.vocab_size()

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, pad_id=0, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def get_pad_id(self):
        return self.char2ind[self.EMPTY_TOK]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        decoded = []
        empty_idx = self.char2ind[self.EMPTY_TOK]
        last_char_idx = empty_idx
        for ind in inds:
            if last_char_idx == ind:
                continue
            if ind != empty_idx:
                decoded.append(self.ind2char[ind])
            last_char_idx = ind
        return "".join(decoded)

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
