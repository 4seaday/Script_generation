import time
import argparse
from tqdm import trange, tqdm
import pandas as pd
from torch.utils.data import DataLoader
import random
import numpy as np
import torch
from datetime import datetime

# import wandb
# wandb.init(project="sbert-medical", entity="4seaday")

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def train(args, retrieval_cls, dataset, output_dir):
    # dataset 정의되면
    # epoch 마다 refence text generation
    
    return


def evaluate(args, retrieval_cls, dataset):
    # PPL compute
    # reference text generation
    
    return acc


def main():
    parser = argparse.ArgumentParser()
    # for setting
    parser.add_argument("--data_path", type=str, default="./data/211020_script_concat.csv")
    parser.add_argument("--model_name", type=str, default="skt/kogpt2-base-v2")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_path", type=str, default="./outputs/")
    
    # tasks
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    
    # for training
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--num_train_epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Training warmup steps")
    
    # for seed setting
    parser.add_argument("--seed", type=int, default=317, help="Training warmup steps")

    args = parser.parse_args()
    set_seed(args)

    # dataset 정의 필요
    dataset = IR_dataset(args)

    if args.do_train:
        

    if args.do_eval:
        
    return

if __name__ == '__main__':
    main()
