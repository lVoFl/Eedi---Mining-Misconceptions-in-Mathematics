import torch
from torch import nn, Tensor
from transformers import AutoModel, PreTrainedTokenizer
from dataclasses import dataclass
from transformers.file_utils import ModelOutput
from typing import Dict, Optional


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None

class BiEncoderModel(nn.Module):
    def __init__(self,
                 model_name: str = None,
                 normlized: bool = False,
                 sentence_pooling_method: str = 'cls',
                 negatives_cross_device: bool = False,
                 temperature: float = 1.0,
                 use_inbatch_neg: bool = True,
                 tokenizer: PreTrainedTokenizer = None
                ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.tokenizer = tokenizer
        assert self.tokenizer is not None
        self.normlized = normlized

    def sentence_embedding(self, hidden_state):
            return hidden_state[:, 0]

    def encode(self, features):
        if features is None:
            return None
        psg_out = self.model(features, return_dict=True)
        p_reps = self.sentence_embedding(psg_out.last_hidden_state)
        if self.normlized:
            p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
        return p_reps.contiguous()

    def compute_loss(self, scores, target):
        # print(scores, target)
        return torch.nn.functional.cross_entropy(scores, target)
        return self.cross_entropy(scores, target)

    def compute_similarity(self, q_reps, p_reps):
        # print(p_reps.shape, q_reps.shape)
        # q_reps, p_reps = q_reps[0], p_reps[0]
        # scores = []
        # for i in range(p_reps.shape[0]):
        #     scores.append(torch.nn.functional.cosine_similarity(q_reps, p_reps[i:i+1]))
        # scores = torch.tensor([scores])
        # print(scores)
        # return scores
        # return torch.nn.functional.cosine_similarity(q_reps[0], p_reps[0])
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def forward(self, query: Tensor, misconceptions: Tensor):
        q_reps = self.encode(query)
        mis_reps = self.encode(misconceptions)
        group_size = mis_reps.size(0) // q_reps.size(0)

        if self.training:
            # scores = self.compute_similarity(q_reps, mis_reps) / self.temperature
            scores = self.compute_similarity(q_reps[:, None, :,], mis_reps.view(q_reps.size(0), group_size, -1)).squeeze(1) / self.temperature
            scores = scores.view(q_reps.size(0), -1)
            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * group_size
            loss = self.compute_loss(scores, target)
            
        else:
            scores = self.compute_similarity(q_reps, mis_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=mis_reps,
        )

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
                 v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)