import pandas as pd
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizer, AutoTokenizer
import datasets
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
        x = self.tokenizer.encode(self.categories[keys], max_length=self.max_len, padding='max_length')
        y = [self.tokenizer.encode(self.misconception[keys], max_length=self.max_len, padding='max_length')]
        a = torch.rand(4)
        a = a*len(self.misconception)
        a = a.long()
        for i in range(a.shape[0]):
            y.append(self.tokenizer.encode(self.misconception[a[i]], max_length=self.max_len, padding='max_length'))
        labels = torch.tensor(0, dtype=torch.long)
        return torch.tensor([x]), torch.tensor(y)
        # return {
        #     'input_ids': (torch.tensor([x]), torch.tensor(y)),
        #     'labels': labels
        # }

@dataclass
class EmbedCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    query_max_len: int = 32
    passage_max_len: int = 128

    def __call__(self, features):
        query = [f[0] for f in features]
        passage = [f[1] for f in features]
        # print(passage)
        query = torch.tensor(query[0])
        passage = torch.tensor(passage[0])
        # print(f'query: {query}')
        # print(len(features))
        # if isinstance(query[0], list):
        #     query = sum(query, [])
        # if isinstance(passage[0], list):
        #     passage = sum(passage, [])

        # q_collated = self.tokenizer(
        #     query,
        #     padding=True,
        #     truncation=True,
        #     max_length=self.query_max_len,
        #     return_tensors="pt",
        # )
        # d_collated = self.tokenizer(
        #     passage,
        #     padding=True,
        #     truncation=True,
        #     max_length=self.passage_max_len,
        #     return_tensors="pt",
        # )
        return {"query": query, "misconceptions": passage}

