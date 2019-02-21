from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
import random

import numpy as np
import torch
from torch import nn

from pytorch_pretrained_bert.modeling import BertForPreTraining, BertForMaskedLM
from pytorch_pretrained_bert.tokenization import BertTokenizer

class LM(nn.Module):
    def __init__(self, bert_model='bert-base-uncased',
          do_lower_case=True, fp16=False,
          local_rank=-1, max_seq_length=128, no_cuda=False,
          seed=42):
        super(LM, self).__init__()

        if local_rank == -1 or no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
            n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend='nccl')

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(seed)

        tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

        # Prepare model
        model = BertForMaskedLM.from_pretrained(bert_model)
        model.eval()

        if fp16:
            model.half()
        model.to(device)
        if local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
            model = DDP(model)
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model)

        self.tokenizer = tokenizer
        self.model = model
        self.no_cuda = no_cuda
        self.max_seq_length = max_seq_length


    def interative(self, times=10):
        for _ in times:
            raw_text = input("Model prompt >>> ")
            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input("Model prompt >>> ")
            ppl = self.get_score(raw_text)
            print("[Info] ppl: {:.2f}".format(ppl))

    def get_dataset_ppl(self, file, printout=''):
        with open(file, 'rb') as f:
            data = [str(line.strip()) for line in f if line.strip()]
        ppl_list = []
        for line in data:
            ppl = self.get_ppl(line)
            ppl_list.append(ppl)
        ppl = np.array(ppl_list).mean()
        if printout:
            print("[Info] {} Dataset ppl: {:.2f}".format(printout, ppl))
        return

    def get_ppl(self, sentence, printout=False):
        tokens = self.tokenizer.tokenize(sentence)
        tokens = tokens[:self.max_seq_length]

        tokens_tensor = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokens)])
        if not self.no_cuda:
            tokens_tensor = tokens_tensor.to('cuda')
        with torch.no_grad():
            loss = self.model(tokens_tensor, masked_lm_labels=tokens_tensor)
            ppl = torch.exp(loss)
            ppl = ppl.item()
        if printout:
            print("[Info] ppl: {:.2f}".format(ppl))

        return ppl


def main():
    lm = LM()
    sentence = 'it is a beautiful day'
    ppl = lm.get_ppl(sentence, printout=True)
    for dataset in ['ag', 'fake', 'yelp', 'mr']:
        file = 'data/{}/train_lm.txt'.format(dataset)
        ppl_list = lm.get_dataset_ppl(file, printout=dataset)
        import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()
