from typing import List, Tuple, Any, Dict, Union

import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset

from mil.data.cache import CacheToRAM


def _h5_loader(path):
    with h5py.File(path, 'r') as f:
        return torch.from_numpy(f['features'][:])

def _pt_loader(path):
    return torch.load(path, weights_only=True, map_location='cpu')

class EmbeddingLoaderFactory:
    _loaders = {
            'h5': _h5_loader,
            'pt': _pt_loader
        }

    @staticmethod
    def get_loader(format_type):
        loader = EmbeddingLoaderFactory._loaders.get(format_type)
        if loader is None:
            raise NotImplementedError(f"{format_type} is not supported.")
        return loader

    @staticmethod
    def list_loaders():
        return list(EmbeddingLoaderFactory._loaders.keys())

# Note that there is no loading of coords value in MIL.
# In inference there will be option to do this along with pointing to the patch/feats h5 file.
class BaseBagDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            id_col: str,
            bag_size: int = None,
            cache: bool = False,
            file_format: str = 'pt',
    ):
        self.__load_function = EmbeddingLoaderFactory.get_loader(file_format)
        self.bag_size = bag_size

        # enforce same order for same datasets
        df = df.sort_values(by=['case_id', 'id']).reset_index(drop=True)
        self.__df = df

        # cache embeddings
        self.__cache_dict = None
        if cache:
            cache = CacheToRAM(
                list(df[['id', 'path']].itertuples(index=False, name=None)),
                self.__load_function,
                workers=8
            )
            cache.cache_data()
            self.__cache_dict = cache.cache_dict

        self.__bags: List[Tuple[str, Any]] = list(df[[id_col, 'target']].drop_duplicates().itertuples(
            index=False, name=None))

    @property
    def df(self):
        return self.__df

    @property
    def bags(self) -> List[Tuple[str, Any]]:
        return self.__bags

    @property
    def targets(self) -> List[Any]:
        return [bag[1] for bag in self.bags]

    @property
    def groups(self) -> Union[List, None]:
        return None

    def __len__(self):
        return len(self.bags)

    def load_slide(self, slide_id):
        if self.__cache_dict is not None:
            return self.__cache_dict[slide_id]

        slide_path = self.df.loc[self.df['id'] == slide_id, 'path'].item()
        return self.__load_function(slide_path)

    def sample_instances(self, bag):
        current_shape = bag.shape[0]
        if current_shape > self.bag_size:
            new_idx = torch.randint(high=current_shape, size=(self.bag_size,))
            bag = bag[new_idx]
        return bag

    def balance(self) -> torch.FloatTensor:
        class_weights = pd.Series(self.targets).value_counts(normalize=True).values
        # TODO implement for multiclass
        weights = class_weights[:1] / class_weights[1:]
        return torch.from_numpy(weights).float()


class SlideBagDataset(BaseBagDataset):
    def __init__(self, df: pd.DataFrame, bag_size: int = None, cache: bool = False, file_format='pt'):
        id_col = 'id'
        super().__init__(df, id_col, bag_size, cache, file_format)

    @property
    def groups(self):
        return self.df['case_id']

    def __getitem__(self, item):
        slide_id, target = self.bags[item]
        embeddings = self.load_slide(slide_id)

        if self.bag_size is not None:
            embeddings = self.sample_instances(embeddings)

        return {
            'id': slide_id,
            'embeddings': embeddings,
            'target': target,
        }

class CaseBagDataset(BaseBagDataset):
    def __init__(self, df: pd.DataFrame, bag_size: int = None, cache: bool = False, file_format='pt'):
        id_col = 'case_id'
        super().__init__(df, id_col, bag_size, cache, file_format)
        self.case_slides_dict: Dict[str, List[str]] = self.df.groupby('case_id')['id'].apply(list).to_dict()

    def __getitem__(self, item):
        case_id, target = self.bags[item]
        embeddings = list()
        for slide_id in self.case_slides_dict[case_id]:
            embeddings.append(self.load_slide(slide_id))
        embeddings = torch.vstack(embeddings)
        if self.bag_size is not None:
            embeddings = self.sample_instances(embeddings)

        return {
            'id': case_id,
            'embeddings': embeddings,
            'slide_ids': None,  # TODO track on slide id locations within bag
            'target': target,
        }
