import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *

class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=150):
        super().__init__()

        self.ffnn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.ffnn(x)
    

class WordEncoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, embed_model, n_layers=2):

        super().__init__()
        self.embed_dim = embed_dim
        self.embed_model = embed_model
        self.lstm = nn.LSTM(embed_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=True,
                            batch_first=True)
        

    def forward(self, inputs):

        embeds = self.embed_words(inputs,self.embed_dim)
        # (L,embed_dim) --> (1, L, embed_dim)
        new_embeds = embeds.unsqueeze(0)

        x, _ = self.lstm(to_cuda(new_embeds))
        # (1, L,embed_dim) --> (L, embed_dim)     
        return embeds, x.squeeze(0)
    
    def embed_words(self, words,embed_size):
        embeddings = np.zeros((len(words),embed_size),dtype='float32')
        for i, word in enumerate(words):
            if word in self.embed_model:
                embeddings[i] = self.embed_model[word]

        return torch.tensor(embeddings)
    

class SpanEncoder(nn.Module):
    def __init__(self,lstm_dim, g_dim):

        super().__init__()
        self.get_alpha = FFNN(lstm_dim)
        self.ffnn = FFNN(g_dim)

    def forward(self, lstm_inputs, embed_inputs,start_idx,end_idx):
        # inputs:(L,N), alpha:(L)
        alphas = self.get_alpha(lstm_inputs)
        
        a_p = F.softmax(alphas[start_idx:end_idx+1,:], dim=1)
        
        x_p = torch.sum(torch.mul(a_p,embed_inputs[start_idx:end_idx+1,:]),dim=0)
        
        g_p = torch.cat((lstm_inputs[start_idx,:], lstm_inputs[end_idx,:],x_p),dim=0)

        gpp = g_p.unsqueeze(0).expand(start_idx,-1)
        inputs = torch.cat((gpp, lstm_inputs[:start_idx]), dim=1)    
        scores = self.ffnn(inputs)

        return scores

class EasySpanEncoder(nn.Module):
    def __init__(self,lstm_dim, g_dim):

        super().__init__()
        self.ffnn = FFNN(g_dim)

    def forward(self, lstm_inputs, embed_inputs,start_idx,end_idx):

        
        x_p = torch.sum(embed_inputs[start_idx:end_idx+1],dim=0)
        x2_p = torch.sum(lstm_inputs[start_idx:end_idx+1],dim=0)

        gpp = x_p.unsqueeze(0).expand(start_idx,-1)
        g2pp = x2_p.unsqueeze(0).expand(start_idx,-1)
        inputs = torch.cat((g2pp,gpp, lstm_inputs[:start_idx]), dim=1)    
        scores = self.ffnn(inputs)

        return scores
    

class CorefModel(nn.Module):
    def __init__(self, embeds_dim,
                       hidden_dim,
                       embed_model,
                       distance_dim=20):

        super().__init__()

        lstm_dim = hidden_dim*2
        g_dim = lstm_dim*3 + embeds_dim
        # g_dim = lstm_dim*2 + embeds_dim
      
        # Initialize modules
        self.encoder = WordEncoder(embeds_dim, hidden_dim, embed_model)
        self.score_spans = SpanEncoder(lstm_dim=lstm_dim,g_dim=g_dim)
        # self.score_spans = EasySpanEncoder(lstm_dim=lstm_dim,g_dim=g_dim)

    def forward(self, sentence, start_idx, end_idx):
        sentence_embed, sentence_lstm = self.encoder(sentence)
        coref_scores = self.score_spans(sentence_lstm, sentence_embed,start_idx,end_idx)
        return coref_scores
    