from torch.utils.data import Dataset
import lmdb
import pickle
from tqdm import tqdm


class LMDBComplexDataset(Dataset):
    def __init__(self, lmdb_path):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with self.env.begin() as txn:
            self.length = pickle.loads(txn.get(b"__len__"))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        key = f"{idx:08d}".encode()
        with self.env.begin() as txn:
            data = pickle.loads(txn.get(key))
        return data
