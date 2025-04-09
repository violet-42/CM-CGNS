import os
import pickle
import re

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from nltk.tokenize import RegexpTokenizer
from cgns.constants import *
from cgns.datasets.utils import get_imgs
from tqdm import tqdm
from transformers import BertTokenizer

# from torch.nn.utils.rnn import pad_sequence

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class MultimodalPretrainingDataset(data.Dataset):
    def __init__(self, split="train", transform=None, data_pct=1.0,
                 imsize=256, max_words=256, sent_num=3):
        super().__init__()
        if not os.path.exists(MIMIC_CXR_DATA_DIR):
            raise RuntimeError(f"{MIMIC_CXR_DATA_DIR} does not exist!")

        self.transform = transform
        self.imsize = imsize
        self.df = pd.read_csv(MIMIC_CXR_MASTER_CSV)
        self.df = self.df[self.df["ViewPosition"].isin(["PA", "AP"])] 
        self.df[MIMIC_CXR_PATH_COL] = self.df[MIMIC_CXR_PATH_COL].apply(
            lambda x: os.path.join(MIMIC_CXR_DATA_DIR, "/".join(x.split("/")[1:])))

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split) 

        self.df = self.df[self.df[MIMIC_CXR_SPLIT_COL] == split] 
        if data_pct != 1.0 and split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42) 
        self.df.reset_index(drop=True, inplace=True) 

        self.tokenizer = BertTokenizer.from_pretrained(
            "/home/***/project/CGNS/Bio_ClinicalBERT/")  
        self.max_words = max_words  

    def load_text_data(self, split):
        # get study to captions mapping
        # TODO: check this
        filepath = os.path.join(
            BASE_DIR, "../../data/captions.pickle")
        if not os.path.isfile(filepath):
            print(
                f"Caption file {filepath} does not exit. Creating captions...")
            path2sent = self.create_path_2_sent_mapping()  
            with open(filepath, "wb") as f:
                pickle.dump(path2sent, f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                path2sent = pickle.load(f) 

        # filter studies to use for current split
        filenames = []
        for row in self.df.itertuples():  
            cur_split = getattr(row, MIMIC_CXR_SPLIT_COL) 
            path = getattr(row, MIMIC_CXR_PATH_COL)
            if cur_split == split and path in path2sent:
                filenames.append(path)

        return filenames, path2sent

    def create_path_2_sent_mapping(self):  
        sent_lens, num_sents = [], []
        path2sent = {} 
        # iterrows is not faster than itertuples ...  but it is ok
        for _, row in tqdm(self.df.iterrows(),
                           total=self.df.shape[0]):  
            # pick impression, findings, //last_paragraph
            captions = ""
            captions += row["impression"]
            captions += " "
            captions += row["findings"]

            # use space instead of newline
            captions = captions.replace("\n", " ") 

            # split sentences
            splitter = re.compile("[0-9]+\.") 
            captions = splitter.split(captions)  
            captions = [point.split(".") for point in captions]  
            captions = [sent for point in captions for sent in point]  

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:
                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())  
                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii") 
                    if len(t) > 0:
                        included_tokens.append(t)

                if len(included_tokens) > 0:  
                    study_sent.append(" ".join(included_tokens)) 

                cnt += len(included_tokens)

            if cnt >= 3:
                sent_lens.append(cnt)  
                num_sents.append(len(study_sent)) 
                path2sent[row[MIMIC_CXR_PATH_COL]] = study_sent  

        # get report word/setence statistics  
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)

        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent

    def __len__(self):
        return len(self.filenames)

    def get_caption(self, path):
        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            raise Exception("no sentence for path")

        # separate different sentences
        series_sents = list(filter(lambda x: x != "", series_sents))  
        sent = " ".join(series_sents)  

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            add_special_tokens=True,  
            truncation=True,
            padding="max_length",
            max_length=self.max_words,
        )  
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])  

        for key, token in tokens.items():  
            tokens[key] = token.squeeze(dim=0)
        num_tokens = x_len
        sentence_index = torch.ones_like(tokens['input_ids']) * -1
        start_index = 1

        # assign the index of sentence to each token
        for idx, sentence in enumerate(series_sents):
            sentnece_token = self.tokenizer(sentence, return_tensors='pt')['input_ids'][0]
            len_sentence = sentnece_token.shape[0] - 2  # Remove [CLS] and [SEP]
            sentence_index[start_index: start_index + len_sentence] = idx + 1
            start_index += len_sentence

        # truncation
        for key, value in tokens.items():
            # print(len(value))
            tokens[key] = value
            # print(len(tokens[key]))
            if len(value) > self.max_words:
                tokens[key] = value[:self.max_words]
        if len(sentence_index) > self.max_words:
            sentence_index = sentence_index[:self.max_words]

        while len(series_sents) < 16:
            series_sents.append("")

        text_meta = {
            'num_tokens': num_tokens,
            'sentence_index': sentence_index,
            'series_sents': series_sents  
        }

        return tokens, text_meta

    # def __getitem__(self, index):
    #     key = self.filenames[index]
    #     text, text_meta = self.get_caption(key)
    #     imgs = get_imgs(key, self.imsize, self.transform, multiscale=False)
    #     batch = {'image': imgs, 'text': text}
    #     if text_meta is not None:
    #         batch['text_meta'] = text_meta
    #     return batch

    def __getitem__(self, index):
        key = self.filenames[index]
        text, text_meta = self.get_caption(key)
        imgs = get_imgs(key, self.imsize, self.transform, multiscale=False)
        batch = {'image': imgs, 'text': text, 'text_meta': text_meta}
        return batch


def multimodal_collate_fn(batch): 
    """sort sequence"""
    imgs, cap_len, ids, tokens, attention = [], [], [], [], []
    path = []
    for b in batch:
        img, cap, cap_l, p = b
        imgs.append(img)
        cap_len.append(cap_l)
        ids.append(cap["input_ids"])
        tokens.append(cap["token_type_ids"])
        attention.append(cap["attention_mask"])
        path.append(p)

    # stack
    imgs = torch.stack(imgs)
    ids = torch.stack(ids).squeeze()
    tokens = torch.stack(tokens).squeeze()
    attention = torch.stack(attention).squeeze()

    # sort and add to dictionary
    sorted_cap_lens, sorted_cap_indices = torch.sort(
        torch.tensor(cap_len), 0, True)

    path = np.array(path)

    return_dict = {
        "caption_ids": ids[sorted_cap_indices],
        "token_type_ids": tokens[sorted_cap_indices],
        "attention_mask": attention[sorted_cap_indices],
        "imgs": imgs[sorted_cap_indices],
        "cap_lens": sorted_cap_lens,
        "path": path[sorted_cap_indices]
    }
    return return_dict


def custom_collate_fn(batch):
    images = [item['image'] for item in batch]
    texts = [item['text'] for item in batch]
    text_metas = [item['text_meta'] for item in batch]

    images = torch.stack(images, dim=0)

    input_ids = torch.stack([t['input_ids'] for t in texts])
    token_type_ids = torch.stack([t['token_type_ids'] for t in texts])
    attention_mask = torch.stack([t['attention_mask'] for t in texts])

    process_texts = {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask
    }

    num_tokens = torch.tensor([meta['num_tokens'] for meta in text_metas])
    sentence_index = torch.stack([meta['sentence_index'] for meta in text_metas])
    text_meta_batched = {
        'num_tokens': num_tokens,
        'sentence_index': sentence_index,
        'series_sents': [meta['series_sents'] for meta in text_metas]
    }

    batched_data = {
        'image': images,
        'text': process_texts,
        'text_meta': text_meta_batched
    }

    return batched_data


if __name__ == "__main__":
    from cgns.datasets.transforms import DataTransforms

    transform = DataTransforms(is_train=True)
    dataset = MultimodalPretrainingDataset(split="train", transform=transform)
    data = dataset[0]
    print(data)
