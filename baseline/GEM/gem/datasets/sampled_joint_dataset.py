import random
import torch


class SampledJointDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets, ratios, max_samples):
        assert len(datasets) == len(ratios), "Each dataset must have a corresponding ratio"
        assert all(r > 0 for r in ratios), "Ratios must be positive"

        self.datasets = datasets
        self.ratios = ratios
        self.total_ratio = sum(ratios)
        self.max_samples = max_samples

        self.indexables = [self._is_indexable(ds) for ds in datasets]
        self.iterators = [iter(ds) if not ix else None for ds, ix in zip(datasets, self.indexables)]

        self.index_map = self._build_index_map(max_samples=max_samples)

        self.length = len(self.index_map)
        print("SampledJointDataset length:", self.length)

    def _is_indexable(self, ds):
        return hasattr(ds, '__getitem__') and hasattr(ds, '__len__')

    def _build_index_map(self, max_samples):
        weights = [r / self.total_ratio for r in self.ratios]
        dataset_ids = random.choices(range(len(self.datasets)), weights=weights, k=max_samples)

        index_map = []

        for ds_id in dataset_ids:
            if self.indexables[ds_id]:
                dataset_length = len(self.datasets[ds_id])
                index = random.randint(0, dataset_length - 1)  # random index WITH replacement
                index_map.append((ds_id, index))
            else:
                index_map.append((ds_id, None))  # Only need ds_id for iterables

        return index_map

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        ds_id, sample_idx = self.index_map[idx]
        ds = self.datasets[ds_id]

        if self.indexables[ds_id]:
            if sample_idx >= len(ds):
                raise IndexError(f"Index {sample_idx} out of range for dataset {ds_id}")
            return ds[sample_idx]
        else:
            try:
                return next(self.iterators[ds_id])
            except StopIteration:
                # Restart the iterator if exhausted
                self.iterators[ds_id] = iter(ds)
                return next(self.iterators[ds_id])

