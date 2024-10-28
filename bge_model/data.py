import pandas as pd
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizer
import numpy as np
import torch
from dataclasses import dataclass

class TrainDatasetForEmbedding(Dataset):
    def __init__(self, df1, df2, tokenizer: PreTrainedTokenizer, max_len=40):
        super().__init__()
        self.df1 = df1
        self.df2 = df2
        self.misconception = []
        self.categories = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        for i in range(len(self.df1)):
            for j in ['A', 'B', 'C', 'D']:
                # MisconceptionDId
                if not np.isnan(self.df1[f'Misconception{j}Id'].iloc[i]):
                    mis_id = self.df1[f'Misconception{j}Id'].iloc[i]
                    SubjectName = self.df1['SubjectName'].iloc[i]
                    ConstructName = self.df1['ConstructName'].iloc[i]
                    text = 'SubjectName: ' + SubjectName + '\n' + ConstructName
                    self.categories.append(text)
                    self.misconception.append(self.df2['MisconceptionName'].iloc[int(mis_id)])

    def __len__(self):
        return len(self.categories)

    def __getitem__(self, keys):
        x = self.tokenizer(
            self.categories[keys],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        y = [self.tokenizer(
            self.misconception[keys],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )]
        a = torch.rand(3) * len(self.misconception)
        a = a.long()
        for i in range(a.shape[0]):
            y.append(self.tokenizer(
                self.misconception[a[i]],
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ))
        labels = torch.tensor(0, dtype=torch.long)
        return x['input_ids'].squeeze(0), torch.stack([yi['input_ids'].squeeze(0) for yi in y]), labels

@dataclass
class EmbedCollator(DataCollatorWithPadding):
    query_max_len: int = 32
    passage_max_len: int = 128

    def __call__(self, features):
        query = [f[0] for f in features]
        passage = [f[1] for f in features]
        query = torch.stack(query)
        passage = torch.cat(passage,dim=0)

        return {
            "query": query,
            "misconceptions": passage,
            # "query_attention_mask": (query != 0).long(),
            # "passage_attention_mask": (passage != 0).long()
        }
