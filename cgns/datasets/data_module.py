import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tqdm import tqdm
import ast


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset, collate_fn, transforms, data_pct, batch_size, num_workers, crop_size=224):
        super().__init__()

        self.dataset = dataset  
        self.collate_fn = collate_fn  
        self.transforms = transforms  
        self.data_pct = data_pct  
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop_size = crop_size 

    def train_dataloader(self):
        if self.transforms:
            transform = self.transforms(True, self.crop_size)
        else:
            transform = None

        dataset = self.dataset(
            split="train", transform=transform, data_pct=self.data_pct)

        # ori_data= dataset[1]


        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
            # prefetch_factor=2
        )

    def val_dataloader(self):
        if self.transforms:
            transform = self.transforms(False, self.crop_size)
        else:
            transform = None
        dataset = self.dataset(
            split="valid", transform=transform, data_pct=self.data_pct)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers
            # prefetch_factor=2
        )

    def test_dataloader(self):
        if self.transforms:
            transform = self.transforms(False, self.crop_size)
        else:
            transform = None
        dataset = self.dataset(
            split="test", transform=transform, data_pct=self.data_pct)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers
            # prefetch_factor=2
        )


class DataModule_pretrain(pl.LightningDataModule):
    def __init__(self, dataset, collate_fn, transforms, data_pct, batch_size, num_workers, crop_size=224,
                 preproc_dir='/home1/Data/MIMIC-CXR-JPJ/mimic-cxr_h5'):
        super().__init__()
        self.dataset = dataset 
        self.collate_fn = collate_fn  
        self.transforms = transforms 
        self.data_pct = data_pct  
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop_size = crop_size  
        self.preproc_dir = preproc_dir  
        os.makedirs(self.preproc_dir, exist_ok=True)  

    def _save_to_hdf5(self, group, data):
        if 'image' in data:
            group.create_dataset('image', data=data['image'].numpy(), dtype=np.float32)

        if 'text' in data:
            text_group = group.create_group('text')
            for key, value in data['text'].items():
                text_group.create_dataset(key, data=value.numpy(), dtype=np.int32)

        if 'text_meta' in data:
            text_meta_group = group.create_group('text_meta')
            for key, value in data['text_meta'].items():
                if isinstance(value, list):
                    text_meta_group.create_dataset(key, data=np.void(str(value).encode('utf-8')))  # 生成字符串,生成字节序列,将这个字节序列包装成 np.void 类型
                elif isinstance(value, int):
                    text_meta_group.create_dataset(key, data=value)
                elif isinstance(value, torch.Tensor):
                    text_meta_group.create_dataset(key, data=value.numpy(), dtype=np.int32)

    def preprocess_and_save(self, split):
        if self.transforms:
            transform = self.transforms(split == "train", self.crop_size)
        else:
            transform = None

        dataset = self.dataset(
            split=split, transform=transform, data_pct=self.data_pct)

        save_file = os.path.join(self.preproc_dir, f'{split}.h5')

        with h5py.File(save_file, 'w') as hf:
            for i in tqdm(range(len(dataset)), desc=f"Processing {split} set"):
                data = dataset[i]
                group = hf.create_group(f'sample_{i}')
                self._save_to_hdf5(group, data)

    def prepare_data_h5(self):
        print("Start preprocessing")
        self.preprocess_and_save('train')
        print("Finished preprocessing train set")
        self.preprocess_and_save('valid')
        print("Finished preprocessing valid set")
        # self.preprocess_and_save('test')
        # print("Finished preprocessing test set")

    def train_dataloader(self):
        return DataLoader(
            HDF5Dataset_pretrain(os.path.join(self.preproc_dir, 'train.h5'), data_pct=self.data_pct),
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            HDF5Dataset_pretrain(os.path.join(self.preproc_dir, 'valid.h5')),
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            HDF5Dataset_pretrain(os.path.join(self.preproc_dir, 'test.h5')),
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


class HDF5Dataset_pretrain(Dataset):

    def __init__(self, hdf5_file, data_pct=1):
        super(HDF5Dataset_pretrain, self).__init__()
        self.hdf5_file = hdf5_file
        self.data_ptc = data_pct
        self.data_indices = self._get_random_indices(data_pct)

    def _get_random_indices(self, data_pct):
        with h5py.File(self.hdf5_file, 'r') as hf:
            total_length = hf['dataset_length'][()]
            # total_length = len(hf.keys())
            print("total_length:{}".format(total_length))
            num_samples = int(total_length * data_pct)
            all_indices = np.arange(total_length)
            if data_pct != 1:
                np.random.shuffle(all_indices)
            return all_indices[:num_samples]

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, index):
        with h5py.File(self.hdf5_file, 'r') as hf:
            real_index = self.data_indices[index]
            sample_group = hf[f'sample_{real_index}']

            image = torch.tensor(sample_group['image'][:], dtype=torch.float32) if 'image' in sample_group else None

            text = {}
            if 'text' in sample_group:
                for key in sample_group['text'].keys():
                    text[key] = torch.tensor(sample_group['text'][key][:], dtype=torch.int32)

            text_meta = {}
            if 'text_meta' in sample_group:
                for key in sample_group['text_meta'].keys():
                    value = sample_group['text_meta'][key][()]
                    if isinstance(value, np.void):
                        text_meta[key] = ast.literal_eval(value.tobytes().decode('utf-8'))
                    elif isinstance(value, np.integer):
                        text_meta[key] = int(value)
                    elif isinstance(value, np.ndarray):
                        text_meta[key] = torch.tensor(value, dtype=torch.int32)

            data = {
                'image': image,
                'text': text,
                'text_meta': text_meta
            }

        return data


class HDF5Dataset_fintune(Dataset):

    def __init__(self, hdf5_file, data_pct=0.1):
        super(HDF5Dataset_fintune, self).__init__()
        self.hdf5_file = hdf5_file
        self.data_indices = self._get_random_indices(data_pct)

    def _get_random_indices(self, data_pct):
        with h5py.File(self.hdf5_file, 'r') as hf:
            total_length = len(hf['data'])
            num_samples = int(total_length * data_pct)
            all_indices = np.arange(total_length)
            np.random.shuffle(all_indices)
            return all_indices[:num_samples]

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, index):
        with h5py.File(self.hdf5_file, 'r') as hf:
            real_index = self.data_indices[index]
            data = hf['data'][real_index]
            label = hf['labels'][real_index]
        return data, label


class DataModule_fintune(pl.LightningDataModule):
    def __init__(self, dataset, collate_fn, transforms, data_pct, batch_size, num_workers, crop_size=224,
                 preproc_dir='/home1/Data/CheXpert/CheXpert-h5/'):  
        super().__init__()
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.transforms = transforms
        self.data_pct = data_pct
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop_size = crop_size
        self.preproc_dir = preproc_dir
        os.makedirs(self.preproc_dir, exist_ok=True)

    def preprocess_and_save(self, split):
        if self.transforms:
            transform = self.transforms(split == "train", self.crop_size)
        else:
            transform = None

        dataset = self.dataset(split=split, transform=None, data_pct=self.data_pct)
        save_file = os.path.join(self.preproc_dir, f'{split}.h5')

        with h5py.File(save_file, 'w') as hf:
            first_item, first_label = dataset[0]
            transformed_first_item = transform(first_item) if transform else first_item
            transformed_shape = transformed_first_item.size() if isinstance(transformed_first_item,
                                                                            torch.Tensor) else np.array(
                transformed_first_item).shape

            hf.create_dataset("data", (len(dataset), *transformed_shape), dtype=np.float32)
            hf.create_dataset("labels", (len(dataset), 5), dtype=np.float32)  # 注意这里的形状

            for i, (image, label) in tqdm(enumerate(dataset), total=len(dataset), desc=f"Processing {split} set"):
                if transform is not None:
                    image = transform(image)

                if isinstance(image, torch.Tensor):
                    image = image.numpy()

                if isinstance(label, torch.Tensor):
                    label = label.numpy()

                hf["data"][i] = image
                hf["labels"][i] = label 

    def prepare_data_h5(self):
        print("start preprosses")
        self.preprocess_and_save('train')
        print("finished preprossing train")
        self.preprocess_and_save('valid')
        print("finished preprossing valid")
        self.preprocess_and_save('test')
        print("finished preprossing test")

    def train_dataloader(self):
        return DataLoader(
            HDF5Dataset_fintune(os.path.join(self.preproc_dir, 'train.h5'), data_pct=self.data_pct),
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            HDF5Dataset_fintune(os.path.join(self.preproc_dir, 'valid.h5')),
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            HDF5Dataset_fintune(os.path.join(self.preproc_dir, 'test.h5')),
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )