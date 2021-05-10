import os
import logging
logging.getLogger().setLevel(logging.CRITICAL)

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader 
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import download, tokenizer, get_tokenizer
from model.torch_gpt2 import GPT2Config, GPT2LMHeadModel
from data import storyDataset
import gluonnlp
from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model

tok_path = get_tokenizer()
model, vocab = get_pytorch_kogpt2_model()
tok = SentencepieceTokenizer(tok_path,  num_best=0, alpha=0)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

data = pd.read_csv('./data/201015_script.csv')
scene_idx = data['scene']
dataset = storyDataset('./data/201015_script.csv', vocab, tok)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

batch_size = 16
epochs = 40
learning_rate = 3e-5
wamup_steps = 5000
max_seq_len = 400

from transformers import AdamW, get_linear_schedule_with_warmup

model = model.to(device)
print("devcie :", device)

model.train()
optimizer = AdamW(model.parameters(), lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps =wamup_steps, num_training_steps=-1)
proc_seq_count = 0
sum_loss = 0.0
batch_count = 0
model.zero_grad()

tmp_line_tens = None
models_folder = 'trained_models'
if not os.path.exists(models_folder):
    os.mkdir(models_folder)

for epoch in range(epochs):
    print(f"Epoch {epoch} started" + '=' * 30)

    for idx, line in enumerate(data_loader):
        line_tens = torch.tensor(line).unsqueeze(0).to(device)
        
        # torch.Size([1, number fo tokens])
        # skip sample from dataset if it is longer than max_seq_len
        if line_tens.size()[1] > max_seq_len:
            continue
        
        # 새로운 씬이 시작될 때.
        if not torch.is_tensor(tmp_line_tens):
            tmp_line_tens = line_tens
            tmp_scene = scene_idx[idx]
            continue
        else:
            if scene_idx[idx] != tmp_scene:
                # 씬넘버 바꾸고 학습
                work_line_tens = tmp_line_tens
                tmp_scene = scene_idx[idx]
                temp_line_tens = line_tens

            elif tmp_line_tens.size()[1] + line_tens.size()[1] > max_seq_len:
                work_line_tens = tmp_line_tens
                tmp_line_tens = line_tens

            else:
                tmp_line_tens = torch.cat([tmp_line_tens, line_tens[:, 1:]], dim=1)
                continue

        # sequence ready, process it through the model
        outputs = model(work_line_tens, labels=work_line_tens)
        loss, logits = outputs[:2]
        loss.backward()
        sum_loss = sum_loss + loss.detach().data

        proc_seq_count = proc_seq_count + 1
        if proc_seq_count == batch_size:
            proc_seq_count = 0
            batch_count += 1
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            model.zero_grad()

        if batch_count == 50:
            print(f"sum loss {sum_loss}")
            batch_count = 0
            sum_loss = 0.0

    # Store the model after each epoch to compare the performance of them
    if epoch % 5 == 0:
        torch.save(model.state_dict(), os.path.join(models_folder, f"201015_gpt2_story_{epoch}.pt"))
        print("model saved.")

