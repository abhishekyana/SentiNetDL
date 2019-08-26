import torch
import torch.nn as nn
import numpy as np
import pickle

class SentimentNet(torch.nn.Module):
    def __init__(self,nE=300,nH=512,nL=2):
        super().__init__()
        self.rnn = torch.nn.LSTM(nE,nH,nL,batch_first=True)
        self.fc = torch.nn.Linear(nH,1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self,X,*kwargs):
        outs,(h,c) = self.rnn(X,*kwargs)
        out = self.fc(h[-1,:,:])#(outs[:,-1,:])
        out = self.sigmoid(out)
        return out

def sentimentnet(pretrained=True,*kwargs):
    SA = SentimentNet()
    if pretrained:
        SA.load_state_dict(torch.load('./SentimentAnalysisModel',map_location='cpu'))
        return SA
    return SA

def GetEmbeddings(path='./SupFiles/GloVe300.d'):
    GloVe = pickle.load(open(path,'rb'))
    W2ID = {w:i for i,w in enumerate(sorted(list(GloVe.keys())))}
    EMB = torch.nn.Embedding(len(W2ID),300)
    EMB.weight.requires_grad=False
    GloVeW = np.vstack([GloVe[w] for w in W2ID])
    EMB.weight.data.copy_(torch.from_numpy(GloVeW))
    return W2ID, EMB


def filterit(s,W2ID):
    s=s.lower()
    S=''
    for c in s:
        if c in ' abcdefghijklmnopqrstuvwxyz0123456789':
            S+=c
    S = " ".join([x  if x and x in W2ID else "<unk>" for x in S.split()])
    return S


def Sentence2Embeddings(sentence,W2ID,EMB):
    if type(sentence)==str:
        sentence = filterit(sentence, W2ID)
        IDS = torch.tensor([W2ID[i] for i in sentence.split(" ")])
        return EMB(IDS)
    if type(sentence)==list:
        sembs = []
        for sent in sentence:
            sent = filterit(sent,W2ID)
            IDS = torch.tensor([W2ID[i] for i in sent.split(" ")])
            sembs.append(EMB(IDS))
        sembs = torch.nn.utils.rnn.pad_sequence(sembs,batch_first=True)
        return sembs




class SentimentAnalyzer:
    def __init__(self, cuda=True):
        self.cuda = cuda
        self.SA = sentimentnet()
        self.W2ID, self.EMB = GetEmbeddings('./SupFiles/GloVe300.d')
        if self.cuda:
             self.SA = self.SA.cuda()
    def __call__(self,sent):
        sembs = Sentence2Embeddings(sent,self.W2ID,self.EMB)
        if self.cuda:
            sembs = sembs.cuda()
        out = (2*(self.SA(sembs))-1).detach().numpy()
        return out
