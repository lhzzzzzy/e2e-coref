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
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.ffnn(x)
    
class Distance(nn.Module):
    """ Learned, continuous representations for: span widths, distance
    between spans
    """

    bins = [1,2,3,4,8,16,32,64]

    def __init__(self, distance_dim=20):
        super().__init__()

        self.dim = distance_dim
        self.embeds = nn.Sequential(
            nn.Embedding(len(self.bins)+1, distance_dim),
            nn.Dropout(0.20)
        )

    def forward(self, *args):
        """ Embedding table lookup """
        return self.embeds(self.stoi(*args))

    def stoi(self, lengths):
        """ Find which bin a number falls into """
        return to_cuda(torch.tensor([
            sum([True for i in self.bins if num >= i]) for num in lengths], requires_grad=False
        ))


class WordEncoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, embed_model, n_layers=1):

        super().__init__()
        self.embed_dim = embed_dim
        self.embed_model = embed_model
        self.lstm = nn.LSTM(embed_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=True,
                            batch_first=True)
        
        self.emb_dropout = nn.Dropout(0.50)
        self.lstm_dropout = nn.Dropout(0.20)

    def forward(self, inputs):

        embeds = self.embed_words(inputs,self.embed_dim)
        self.emb_dropout(embeds[0])
        # (L,embed_dim) --> (1, L, embed_dim)
        new_embeds = embeds.unsqueeze(0)

        x, _ = self.lstm(to_cuda(new_embeds))
        # (1, L,embed_dim) --> (L, embed_dim)   
        self.lstm_dropout(x[0])     
        return embeds, x.squeeze(0)
    
    def embed_words(self, words,embed_size):
        embeddings = np.zeros((len(words),embed_size),dtype='float32')
        for i, word in enumerate(words):
            if word in self.embed_model:
                embeddings[i] = self.embed_model[word]

        return torch.tensor(embeddings)
    

class SpanEncoder(nn.Module):
    def __init__(self,lstm_dim, distant_dim, gi_dim, gij_dim):

        super().__init__()
        self.get_alpha = FFNN(lstm_dim)
        self.ffnn_m = FFNN(gi_dim)
        self.ffnn_a = FFNN(gij_dim)
        self.width = Distance()
        self.distance = Distance()

    def forward(self, lstm_inputs, embed_inputs):
        # inputs:(L,N), alpha:(L)
        alphas = self.get_alpha(lstm_inputs)


        spans = self.get_spans(4,lstm_inputs.size(0))

        span_alphas = [alphas[s["start"]: s["end"] +1] for s in spans]
        span_embeds = [embed_inputs[s["start"]: s["end"]+1] for s in spans]
        


        #这里padded_attns使用-1e10作为补全的内容，因为-1e10在softmax过程中可以被忽略。-1e10看作负无穷。
        padded_attns, _ = pad_and_stack(span_alphas, value=-1e10) 
        padded_embeds, _ = pad_and_stack(span_embeds)



        a_it = F.softmax(padded_attns,dim=1)

        
        attn_embeds = torch.sum(torch.mul(to_cuda(padded_embeds), to_cuda(a_it)), dim=1)

        # Compute span widths (i.e. lengths), embed them
        widths = self.width([ (s["end"]-s["start"]) for s in spans])


        start_end = torch.stack([torch.cat((lstm_inputs[s["start"]], lstm_inputs[s["end"]])) for s in spans])


        g_i = torch.cat((start_end,attn_embeds,widths),dim=1)


        mention_score = self.ffnn_m(g_i)

        for i in range(len(spans)):
            spans[i]["mscore"] = mention_score[i]

        spans = self.purne_and_get_prespan(spans, embed_inputs.size(0))

        gidx = []
        for i in range(len(spans)):
            gidx.append(spans[i]["idx"])

        g_ij = []
        idx_i,idx_j,dis = [],[],[]

        for i in range(1, len(spans)):
            for j in range(i):
                idx_i.append(spans[i]["idx"])
                idx_j.append(spans[j]["idx"])
                dis.append(spans[i]["start"] - spans[j]["start"])

        phi_ij = self.distance(dis)

        gi = torch.index_select(to_cuda(g_i),0,to_cuda(torch.tensor(idx_i)))
        gj = torch.index_select(to_cuda(g_i),0,to_cuda(torch.tensor(idx_j)))


        g_ij = torch.cat((gi,gj,gi*gj,phi_ij),dim=1)

        antecedent_score = self.ffnn_a(g_ij)

        sm_i = torch.index_select(to_cuda(mention_score), 0, to_cuda(torch.tensor(idx_i)))
        sm_j = torch.index_select(to_cuda(mention_score), 0, to_cuda(torch.tensor(idx_j)))

        coref_scores = torch.sum(torch.cat((sm_i, sm_j, antecedent_score), dim=1), dim=1, keepdim=True)

        antecedent_idx = [len(span["pre_spans"]) for span in spans]

        # Split coref scores so each list entry are scores for its antecedents, only.
        # (NOTE that first index is a special case for torch.split, so we handle it here)
        split_scores =  list(torch.split(coref_scores, antecedent_idx, dim=0))

        epsilon = to_var(torch.tensor([[0.]]))
        with_epsilon = [torch.cat((score, epsilon), dim=0) for score in split_scores]

        probs = [F.softmax(tensr,dim=0) for tensr in with_epsilon]
        
        

        # pad the scores for each one with a dummy value, 1000 so that the tensors can 
        # be of the same dimension for calculation loss and what not. 
        probs, _ = pad_and_stack(probs, value=1000)
        probs = probs.squeeze()

        return spans, probs

    def get_spans(self, L, n):
        spans = []
        num = 0
        for start in range(n):
            for end in range(start, min(start + L, n)):
                spans.append({
                    "start":start, 
                    "end":end, 
                    "mscore":0,
                    "pre_spans": [],
                    "idx":num})
                num += 1
        return spans
    
    def purne_and_get_prespan(self, spans, T, LAMBDA=0.4):
        new_spans = []
        """ Prune mention scores to the top lambda percent.
            Returns list of tuple(scores, indices, g_i) """

        # Only take top λT spans, where T = len(doc)
        STOP = int(LAMBDA * T)

        # Sort by mention score, remove overlapping spans, prune to top λT spans
        sorted_spans = sorted(spans, key=lambda s: s["mscore"], reverse=True)

        nonoverlapping, seen = [], set()
        for span in sorted_spans:
            flag = True
            for (st,ed) in seen:
                if (span["start"]<st and st<= span["end"] and span["end"]< ed) or (st < span["start"] and span["start"]<=ed and ed <span["end"]):
                    flag = False
                    break
                
            if flag:
                nonoverlapping.append(span)
                seen.add((span["start"],span["end"]))              

        pruned_spans = nonoverlapping[:STOP]

        # Resort by start, end indexes
        spans = sorted(pruned_spans, key=lambda s: (s["start"], s["end"]))

        for i in range(1,len(spans)):
            spans[i]["pre_spans"] = [t for t in range(i)]

        return spans
    

class CorefModel(nn.Module):
    def __init__(self, embeds_dim,
                       hidden_dim,
                       embed_model,
                       distance_dim=20):

        super().__init__()

        # Forward and backward pass over the document
        lstm_dim = hidden_dim*2

        # Forward and backward passes, avg'd attn over embeddings, span width
        gi_dim = lstm_dim*2 + embeds_dim + distance_dim

        # gi, gj, gi*gj, distance between gi and gj
        gij_dim = gi_dim*3 + distance_dim

        # Initialize modules
        self.encoder = WordEncoder(embeds_dim, hidden_dim, embed_model)
        self.score_spans = SpanEncoder(lstm_dim=lstm_dim,distant_dim=distance_dim,gi_dim=gi_dim,gij_dim=gij_dim)

    def forward(self, sentence):
        sentence_embed, sentence_lstm = self.encoder(sentence)
        spans, coref_scores = self.score_spans(sentence_lstm, sentence_embed)
        return spans, coref_scores
    