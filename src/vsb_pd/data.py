from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import PipelineConfig


class WindowDataset(Dataset):
    """Loads NPZ windows plus labels from a development manifest."""

    def __init__(self, manifest_path: Path, config: PipelineConfig):
        manifest = pd.read_parquet(manifest_path)
        self.artifact_paths = [Path(p) for p in manifest["artifact_path"]]
        metadata = pd.read_csv(config.metadata_path)
        metadata_by_id = metadata.set_index("id_measurement")
        self.labels: list[np.ndarray] = []
        self.measurement_ids: list[int] = []
        for artifact_path in self.artifact_paths:
            with np.load(artifact_path, allow_pickle=False) as a:
                mid = int(a["measurement_id"].item())
                self.measurement_ids.append(mid)
            group = metadata_by_id.loc[mid]
            if isinstance(group, pd.Series):
                self.labels.append(group["target"].astype(np.int8).to_numpy() if hasattr(group, 'to_numpy') else np.array([group["target"]], dtype=np.int8))
            else:
                self.labels.append(group["target"].to_numpy(dtype=np.int8))

    def __len__(self) -> int:
        return len(self.artifact_paths)

    def __getitem__(self, idx: int):
        with np.load(self.artifact_paths[idx], allow_pickle=False) as a:
            windows = torch.from_numpy(a["windows"].copy())
            starts = torch.from_numpy(a["starts"].copy())
            kinds = torch.from_numpy(a["kinds"].copy())
            scores = torch.from_numpy(a["scores"].copy())
        targets = torch.tensor(self.labels[idx].tolist(), dtype=torch.float32)
        return windows, starts, kinds, scores, targets, self.measurement_ids[idx]
