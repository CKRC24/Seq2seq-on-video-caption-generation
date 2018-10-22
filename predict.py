import numpy as np
import pickle
import pandas as pd
import json
import math
import torch
import string
from random import randint
from collections import defaultdict, Counter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import pickle
import random
import sys

use_cuda = torch.cuda.is_available()
print(use_cuda)

DATA_DIR = sys.argv[1]
output_file = sys.argv[2]

# file path 
test_id_file = DATA_DIR + '/testing_id.txt'

MAXLEN=21
word2index = pickle.load(open("hw2_1_w2i", "rb"))
index2word = pickle.load(open("hw2_1_i2w", "rb"))
VOCAB_SIZE = len(word2index)
print("vocab size: %d" % (VOCAB_SIZE))


class Attn(nn.Module):
    def __init__(self, batch_size, hidden_size, dropout=0.3):
        super(Attn, self).__init__()
        self.nn_attn = nn.Linear(hidden_size*2, 1)
        self.dropout = nn.Dropout(p=dropout)
        self.hidden_size = hidden_size
        
    def forward(self, mode, z, encoder_outputs):
        if mode == "nn":
            # (128, 80, 256)
            dup_z = z.transpose(0,1).expand(z.size()[1], 80, self.hidden_size)
            attn_output = self.nn_attn(torch.cat((encoder_outputs.transpose(0,1), dup_z), 2))
        elif mode == "dot":
            # (128, 80, 256) * (128, 256, 1) = (128, 80, 1)
            attn_output = torch.bmm(encoder_outputs.transpose(0,1), z.transpose(0,1).transpose(1,2))
        attn_output = F.tanh(attn_output)
        attn_weights = F.softmax(attn_output, dim=1)
        return attn_weights
    
class S2VT(nn.Module):
    def __init__(self, feature_size,vocab_size,hidden_size,video_step,output_step,batch_size,n_layers=1,dropout=0.3):
        super(S2VT, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.feature_size = feature_size
        self.embedding_size = 512
        self.video_step = video_step
        self.output_step = output_step
        
        self.attn = Attn(batch_size, hidden_size)
        self.gru1 = nn.GRU(512, hidden_size, n_layers, dropout=dropout)
        self.gru2 = nn.GRU(hidden_size*2+self.embedding_size, hidden_size, n_layers, dropout=dropout)
        self.embedding = nn.Embedding(vocab_size, self.embedding_size)
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(feature_size, 512)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, video_seq, cap_seq, teacher_forcing_ratio):
        loss = 0
        # pad MAXLEN, batch, 4096
        padding_gru1 = Variable(torch.zeros(self.output_step, self.batch_size, 512))
        # pad 80, batch, 256
        padding_gru2 = Variable(torch.zeros(self.video_step, self.batch_size, self.hidden_size+self.embedding_size))
        init_BOS = [0] * self.batch_size
        init_BOS = Variable(torch.LongTensor([init_BOS])).resize(batch_size, 1)
        init_BOS = self.embedding(init_BOS)

        if use_cuda:
            padding_gru1.cuda()
            padding_gru2.cuda()
            init_BOS.cuda()
        
        video_seq = self.dropout(F.selu(self.fc1(video_seq)))
        
        gru1_input = torch.cat((video_seq, padding_gru1), 0)
        # output1:  (seq_len, batch, hidden_size)
        output1, hidden1 = self.gru1(gru1_input)
        
        # cap_seq: batch, MAXLEN => batch, MAXLEN, hidden_size
        embedded = self.embedding(cap_seq)
#         embedded = F.selu(embedded)
        gru2_input = torch.cat((padding_gru2, output1[:self.video_step,:,:]),2)
        output2, decoder_hidden = self.gru2(gru2_input)
        z = decoder_hidden
        # decoder
        for step in range(self.output_step):
            use_teacher_forcing = True if random.random() <= teacher_forcing_ratio else False
            if step == 0:
                decoder_input = init_BOS
            elif use_teacher_forcing:
                decoder_input = embedded[:,step-1,:].unsqueeze(1)
            else:
                decoder_input = decoder_output.max(1)[-1].resize(batch_size, 1)
                decoder_input = self.embedding(decoder_input)
            
            attn_weights = self.attn('dot', z, output1[:self.video_step])
            c = torch.bmm(attn_weights.transpose(1,2),
                                 output1[:self.video_step].transpose(0,1))
            # batch, 1, hidden_size*2
            gru2_input = torch.cat((decoder_input, output1[self.video_step+step].unsqueeze(1), c),2).transpose(0,1)
            
            decoder_output, z = self.gru2(gru2_input, z)
            decoder_output = self.softmax(self.out(decoder_output[0]))
            
            loss += F.nll_loss(decoder_output, cap_seq[:,step])
        return loss
    
    def testing(self, video_seq, index2word, use_beam_search, beam_size):
        pred = []
        padding_gru1 = Variable(torch.zeros(self.output_step, 1, 512)).cuda()
        padding_gru2 = Variable(torch.zeros(self.video_step, 1, self.hidden_size+self.embedding_size)).cuda()
        init_BOS = [0]
        init_BOS = Variable(torch.LongTensor([init_BOS])).cuda().resize(1, 1)
        init_BOS = self.embedding(init_BOS)

        
        video_seq = F.selu(self.fc1(video_seq))
        
        gru1_input = torch.cat((video_seq, padding_gru1), 0)
        output1, hidden1 = self.gru1(gru1_input)
        
        gru2_input = torch.cat((padding_gru2, output1[:self.video_step,:,:]),2)
        output2, decoder_hidden = self.gru2(gru2_input)
        z = decoder_hidden
        
        if use_beam_search:
            for step in range(self.output_step):
                if step == 0:
                    attn_weights = self.attn('dot', z, output1[:self.video_step])
                    c = torch.bmm(attn_weights.transpose(1,2),
                                         output1[:self.video_step].transpose(0,1))
                    # batch, 1, hidden_size*2
                    gru2_input = torch.cat((init_BOS, output1[self.video_step+step].unsqueeze(1), c),2).transpose(0,1)

                    decoder_output, z = self.gru2(gru2_input, z)
                    decoder_output = self.softmax(self.out(decoder_output[0]))

                    softmax_prob = math.e ** decoder_output
                    top_cand_val, top_cand_ix = softmax_prob.topk(beam_size)
                    cur_scores = top_cand_val.data[0].cpu().numpy().tolist()
                    candidates = top_cand_ix.data[0].cpu().numpy().reshape(beam_size, 1).tolist()
                    zs = [z] * beam_size
                else:
                    new_candidates = []
                    for j, candidate in enumerate(candidates):
                        decoder_input = Variable(torch.LongTensor([candidate[-1]])).cuda().resize(1,1)
                        decoder_input = self.embedding(decoder_input)
                        
                        attn_weights = self.attn('dot', z, output1[:self.video_step])
                        c = torch.bmm(attn_weights.transpose(1,2),
                                             output1[:self.video_step].transpose(0,1))
                        # batch, 1, hidden_size*2
                        gru2_input = torch.cat((decoder_input, output1[self.video_step+step].unsqueeze(1), c),2).transpose(0,1)
                        decoder_output, zs[j] = self.gru2(gru2_input, zs[j])
                        decoder_output = self.softmax(self.out(decoder_output[0]))
                        
                        softmax_prob = math.e ** decoder_output
                        top_cand_val, top_cand_ix = softmax_prob.topk(beam_size)
                        for k in range(beam_size):
                            score = cur_scores[j] * top_cand_val.data[0, k]
                            new_candidate = candidates[j] + [top_cand_ix.data[0, k]]
                            new_candidates.append([score, new_candidate, zs[j]])
                    # get top-k candidates and drop others
                    new_candidates = sorted(new_candidates, key=lambda x: x[0], reverse=True)[:beam_size]
                    cur_scores = [candi[0] for candi in new_candidates]
                    candidates = [candi[1] for candi in new_candidates]
                    zs = [candi[2] for candi in new_candidates]
            pred = [index2word[wix] for wix in candidates[0] if wix >= 3]
        else:
            for step in range(self.output_step):
                if step == 0:
                    decoder_input = init_BOS
                else:
                    decoder_input = decoder_output.max(1)[-1].resize(1, 1)
                    decoder_input = self.embedding(decoder_input)
                attn_weights = self.attn('dot', z, output1[:self.video_step])
                c = torch.bmm(attn_weights.transpose(1,2),
                                     output1[:self.video_step].transpose(0,1))
                # batch, 1, hidden_size*2
                gru2_input = torch.cat((decoder_input, output1[self.video_step+step].unsqueeze(1), c),2).transpose(0,1)

                decoder_output, z = self.gru2(gru2_input, z)
                decoder_output = self.softmax(self.out(decoder_output[0]))
                output = decoder_output.max(1)[-1].resize(1, 1)
                word2ix = output.data[0,0]
                ix2word = index2word[word2ix]
                if word2ix < 3:
                    break
                else:
                    pred.append(ix2word)

        return pred
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def inverse_sigmoid(k, i):
    return k / (k + math.exp(i / k))

hidden_size = 512
batch_size = 64
feature_size = 4096
seq_len = 80
s2vt_model = S2VT(feature_size,VOCAB_SIZE,hidden_size,seq_len,MAXLEN,batch_size)
s2vt_model.load_state_dict(torch.load("s2vt_model"))
s2vt_opti = optim.Adam(s2vt_model.parameters(), lr = 0.0003)
if use_cuda:
    s2vt_model.cuda()
print("S2VT model parameters count: %d" % (count_parameters(s2vt_model)))

test_frames = {}
test_label = pd.read_fwf(test_id_file, header=None)
for index, row in test_label.iterrows():
    test_file = DATA_DIR + "/feat/" + row[0] + ".npy"
    test_frames[row[0]] = torch.FloatTensor(np.load(test_file))

s2vt_model.eval()
predictions = []
indices = []
use_beam_search = True
beam_size = 2
for i, row in test_label.iterrows():
    video_input = Variable(test_frames[row[0]].view(-1, 1, feature_size)).cuda()
    pred = s2vt_model.testing(video_input, index2word, use_beam_search, beam_size)
    pred[0] = pred[0].title()
    pred = " ".join(pred)
    predictions.append(pred)
    indices.append(row[0])
    # print(row[0] + " / " + pred)

with open(output_file, 'w') as result_file:
    for i in range(100):
        result_file.write(indices[i] + "," + predictions[i] + "\n")
