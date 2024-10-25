import torch
from transformers import AutoTokenizer, TrainingArguments
from modeling import BiEncoderModel
from data import TrainDatasetForEmbedding, EmbedCollator
from trainer import BiTrainer
import pandas as pd

# 加载模型和 tokenizer
model_name = "BAAI/bge-large-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BiEncoderModel(model_name=model_name, normlized=True, tokenizer=tokenizer)
model.train()

# 加载数据
train_data = pd.read_csv('./eedi-mining-misconceptions-in-mathematics/train.csv')
misconception = pd.read_csv('./eedi-mining-misconceptions-in-mathematics/misconception_mapping_utf-8.csv')
data = TrainDatasetForEmbedding(train_data, misconception, tokenizer)
data_collator = EmbedCollator(
    tokenizer=tokenizer,
    query_max_len=32,        # 设置为适合您的数据的最大长度
    passage_max_len=128
)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./bge_model/model",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    save_steps=200,
    save_total_limit=3,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=5e-5
)

# 初始化 Trainer
trainer = BiTrainer(
    model=model,
    args=training_args,
    train_dataset=data,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# 开始训练
trainer.train()
