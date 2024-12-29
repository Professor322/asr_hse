import json
import os
import shutil
from pathlib import Path

import torchaudio
import wget
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH

URL_LINKS_DATA = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "dev-other": "https://www.openslr.org/resources/12/dev-other.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
}

URL_LINKS_LANGUAGE_MODELS = {
    "3-gram.arpa": "https://www.openslr.org/resources/11/3-gram.arpa.gz",
    "3-gram.pruned.1e-7.arpa": "https://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz",
    "3-gram.pruned.3e-7.arpa": "https://www.openslr.org/resources/11/3-gram.pruned.3e-7.arpa.gz",
    "4-gram.arpa": "https://openslr.elda.org/resources/11/4-gram.arpa.gz",
}


class LibrispeechDataset(BaseDataset):
    def __init__(self, part, data_dir=None, lm_model=None, *args, **kwargs):
        assert part in URL_LINKS_DATA or part == "train_all"

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "librispeech"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        if part == "train_all":
            index = sum(
                [
                    self._get_or_load_index(part)
                    for part in URL_LINKS_DATA
                    if "train" in part
                ],
                [],
            )
        else:
            index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)
        # TODO handle train-all
        data_file_path = self._data_dir / f"{part}_index.json"
        vocab_file_path = self._data_dir / f"{part}_vocab.json"
        sp_model_prefix = self._data_dir / f"bpe_tokenizer"

        lm_model_path = str(self._data_dir / lm_model)
        # check if model is already downloaded
        if not os.path.isfile(lm_model_path):
            lm_model_path = None

        # if it is not downloaded, then download it
        if lm_model is not None and lm_model_path is None:
            lm_model_arch_path = self._data_dir / f"{lm_model}.gz"
            wget.download(URL_LINKS_LANGUAGE_MODELS[lm_model], str(lm_model_arch_path))
            os.system(f"gunzip {lm_model_arch_path}")
            os.remove(str(lm_model_arch_path))
            lm_model_path = str(self._data_dir / lm_model)

        self.text_encoder.setup(
            data_file_path=str(data_file_path),
            vocab_file_path=str(vocab_file_path),
            sp_model_prefix=str(sp_model_prefix),
            lm_model_path=lm_model_path,
        )

    def _load_part(self, part):
        arch_path = self._data_dir / f"{part}.tar.gz"
        print(f"Loading part {part}")
        wget.download(URL_LINKS_DATA[part], str(arch_path))
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "LibriSpeech").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LibriSpeech"))

    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        split_dir = self._data_dir / part
        if not split_dir.exists():
            self._load_part(part)

        flac_dirs = set()
        for dirpath, dirnames, filenames in os.walk(str(split_dir)):
            if any([f.endswith(".flac") for f in filenames]):
                flac_dirs.add(dirpath)
        for flac_dir in tqdm(
            list(flac_dirs), desc=f"Preparing librispeech folders: {part}"
        ):
            flac_dir = Path(flac_dir)
            trans_path = list(flac_dir.glob("*.trans.txt"))[0]
            with trans_path.open() as f:
                for line in f:
                    f_id = line.split()[0]
                    f_text = " ".join(line.split()[1:]).strip()
                    flac_path = flac_dir / f"{f_id}.flac"
                    t_info = torchaudio.info(str(flac_path))
                    length = t_info.num_frames / t_info.sample_rate
                    index.append(
                        {
                            "path": str(flac_path.absolute().resolve()),
                            "text": f_text.lower(),
                            "audio_len": length,
                        }
                    )
        return index
