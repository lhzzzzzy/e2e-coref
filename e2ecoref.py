import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from datetime import datetime
from tqdm import tqdm
import networkx as nx
from subprocess import Popen


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
        return torch.tensor([
            sum([True for i in self.bins if num >= i]) for num in lengths], requires_grad=False
        )


class WordEncoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, n_layers=1):

        super().__init__()
        self.glove = None
        self.lstm = nn.LSTM(embed_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=True,
                            batch_first=True)

    def forward(self, inputs):

        embeds = self.embed_model(inputs)
        # (L,embed_dim) --> (1, L, embed_dim)
        embeds = embeds.unsqueeze(0)

        x, _ = self.lstm(embeds)
        # (1, L,embed_dim) --> (L, embed_dim)        
        return embeds.squeeze(0), x.squeeze(0)
    
    def embed_model(self, inputs):
        return torch.zeros(20, 10) 
    

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

        
        attn_embeds = torch.sum(torch.mul(padded_embeds, a_it), dim=1)

        # Compute span widths (i.e. lengths), embed them
        widths = self.width([ (s["end"]-s["start"]) for s in spans])


        start_end = torch.stack([torch.cat((lstm_inputs[s["start"]], lstm_inputs[s["end"]])) for s in spans])


        g_i = torch.cat((start_end,attn_embeds,widths),dim=1)


        mention_score = self.ffnn_m(g_i)

        for i in range(len(spans)):
            spans[i]["mscore"] = mention_score[i]
        # TODO:
        spans = self.purne_and_get_prespan(spans, 20)

        gidx = []
        for i in range(len(spans)):
            gidx.append(spans[i]["idx"])

        g_ij = []
        idx_i,idx_j,dis = [],[],[]
        # TODO:
        for i in range(1, len(spans)):
            for j in range(i):
                idx_i.append(spans[i]["idx"])
                idx_j.append(spans[j]["idx"])
                dis.append(spans[i]["start"] - spans[j]["start"])

        phi_ij = self.distance(dis)

        gi = torch.index_select(g_i,0,torch.tensor(idx_i))
        gj = torch.index_select(g_i,0,torch.tensor(idx_j))


        g_ij = torch.cat((gi,gj,gi*gj,phi_ij),dim=1)

        antecedent_score = self.ffnn_a(g_ij)

        sm_i = torch.index_select(mention_score, 0, torch.tensor(idx_i))
        sm_j = torch.index_select(mention_score, 0, torch.tensor(idx_j))

        coref_scores = torch.sum(torch.cat((sm_i, sm_j, antecedent_score), dim=1), dim=1, keepdim=True)

        antecedent_idx = [len(span["pre_spans"]) for span in spans]

        # Split coref scores so each list entry are scores for its antecedents, only.
        # (NOTE that first index is a special case for torch.split, so we handle it here)
        split_scores = [to_cuda(torch.tensor([]))] \
                         + list(torch.split(coref_scores, antecedent_idx, dim=0))

        epsilon = to_var(torch.tensor([[0.]]))
        with_epsilon = [torch.cat((score, epsilon), dim=0) for score in split_scores]

        probs = [F.softmax(tensr) for tensr in with_epsilon]
        
        

        # pad the scores for each one with a dummy value, 1000 so that the tensors can 
        # be of the same dimension for calculation loss and what not. 
        probs, _ = pad_and_stack(probs, value=1000)
        probs = probs.squeeze()

        return spans, probs

    def get_spans(self, L, n):
        spans = []
        num = 0
        for start in range(n):
            for end in range(start + 1, min(start + L, n)):
                spans.append({
                    "start":start, 
                    "end":end, 
                    "mscore":0,
                    "pre_spans": [],
                    "idx":num})
                num += 1
        return spans
    
    def purne_and_get_prespan(self, spans, T, LAMBDA=0.5):
        new_spans = []
        """ Prune mention scores to the top lambda percent.
            Returns list of tuple(scores, indices, g_i) """

        # Only take top λT spans, where T = len(doc)
        STOP = int(LAMBDA * T)

        # Sort by mention score, remove overlapping spans, prune to top λT spans
        sorted_spans = sorted(spans, key=lambda s: s["mscore"], reverse=True)

        nonoverlapping, seen = [], set()
        for s in sorted_spans:
            indexes = range(s["start"], s["end"]+1)
            taken = [i in seen for i in indexes]
            if len(set(taken)) == 1 or (taken[0] == taken[-1] == False):
                nonoverlapping.append(s)
                seen.update(indexes)

        pruned_spans = nonoverlapping[:STOP]

        # Resort by start, end indexes
        spans = sorted(pruned_spans, key=lambda s: (s["start"], s["end"]))

        for i in range(1,len(spans)):
            spans[i]["pre_spans"] = [t for t in range(i)]

        return spans
    

class CorefModel(nn.Module):
    def __init__(self, embeds_dim,
                       hidden_dim,
                       char_filters=50,
                       distance_dim=20,
                       genre_dim=20,
                       speaker_dim=20):

        super().__init__()

        # Forward and backward pass over the document
        lstm_dim = hidden_dim*2

        # Forward and backward passes, avg'd attn over embeddings, span width
        gi_dim = lstm_dim*2 + embeds_dim + distance_dim

        # gi, gj, gi*gj, distance between gi and gj
        gij_dim = gi_dim*3 + distance_dim

        # Initialize modules
        self.encoder = WordEncoder(embeds_dim, hidden_dim)
        self.score_spans = SpanEncoder(lstm_dim=lstm_dim,distant_dim=distance_dim,gi_dim=gi_dim,gij_dim=gij_dim)

    def forward(self, sentence):
        sentence_embed, sentence_lstm = self.encoder(sentence)
        spans, coref_scores = self.score_spans(sentence_lstm, sentence_embed)
        return spans, coref_scores
    


class Trainer:
    """ Class dedicated to training and evaluating the model
    """
    def __init__(self, model, train_corpus, val_corpus, test_corpus,
                    lr=1e-3, steps=100):

        self.__dict__.update(locals())

        self.train_corpus = list(self.train_corpus)
        self.val_corpus = self.val_corpus
        
        self.model = to_cuda(model)

        self.optimizer = optim.Adam(params=[p for p in self.model.parameters()
                                            if p.requires_grad], lr=lr)

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=100,
                                                   gamma=0.001)

    def train(self, num_epochs, eval_interval=10, *args, **kwargs):
        """ Train a model """
        for epoch in range(1, num_epochs+1):
            self.train_epoch(epoch, *args, **kwargs)

            # Save often
            self.save_model(str(datetime.now()))

            # Evaluate every eval_interval epochs
            if epoch % eval_interval == 0:
                print('\n\nEVALUATION\n\n')
                self.model.eval()
                results = self.evaluate(self.val_corpus)
                print(results)

    def train_epoch(self, epoch):
        """ Run a training epoch over 'steps' documents """

        # Set model to train (enables dropout)
        self.model.train()

        # Randomly sample documents from the train corpus
        # batch = random.sample(self.train_corpus, self.steps)
        batch = self.train_corpus

        epoch_loss, epoch_mentions, epoch_corefs, epoch_identified = [], [], [], []

        for document in tqdm(batch):

            # Randomly truncate document to up to 50 sentences
            doc = document.truncate()

            # Compute loss, number gold links found, total gold links
            loss, mentions_found, total_mentions, \
                corefs_found, total_corefs, corefs_chosen = self.train_doc(doc)

            # Track stats by document for debugging
            print(document, '| Loss: %f | Mentions: %d/%d | Coref recall: %d/%d | Corefs precision: %d/%d' \
                % (loss, mentions_found, total_mentions,
                    corefs_found, total_corefs, corefs_chosen, total_corefs))

            epoch_loss.append(loss)
            epoch_mentions.append(safe_divide(mentions_found, total_mentions))
            epoch_corefs.append(safe_divide(corefs_found, total_corefs))
            epoch_identified.append(safe_divide(corefs_chosen, total_corefs))

        # Step the learning rate decrease scheduler
        self.scheduler.step()

        print('Epoch: %d | Loss: %f | Mention recall: %f | Coref recall: %f | Coref precision: %f' \
                % (epoch, np.mean(epoch_loss), np.mean(epoch_mentions),
                    np.mean(epoch_corefs), np.mean(epoch_identified)))

    def train_doc(self, document):
        """ Compute loss for a forward pass over a document """

        # Extract gold coreference links
        gold_corefs, total_corefs, \
            gold_mentions, total_mentions = extract_gold_corefs(document)

        # Zero out optimizer gradients
        self.optimizer.zero_grad()

        # Init metrics
        mentions_found, corefs_found, corefs_chosen = 0, 0, 0

        # Predict coref probabilites for each span in a document
        spans, probs = self.model(document)

        # Get log-likelihood of correct antecedents implied by gold clustering
        gold_indexes = to_cuda(torch.zeros_like(probs))
        for idx, span in enumerate(spans):

            # Log number of mentions found
            if (span.i1, span.i2) in gold_mentions:
                mentions_found += 1

                # Check which of these tuples are in the gold set, if any
                golds = [
                    i for i, link in enumerate(span.yi_idx)
                    if link in gold_corefs
                ]

                # If gold_pred_idx is not empty, consider the probabilities of the found antecedents
                if golds:
                    gold_indexes[idx, golds] = 1

                    # Progress logging for recall
                    corefs_found += len(golds)
                    found_corefs = sum((probs[idx, golds] > probs[idx, len(span.yi_idx)])).detach()
                    corefs_chosen += found_corefs.item()
                else:
                    # Otherwise, set gold to dummy
                    gold_indexes[idx, len(span.yi_idx)] = 1

        # Negative marginal log-likelihood
        eps = 1e-8
        loss = torch.sum(torch.log(torch.sum(torch.mul(probs, gold_indexes), dim=1).clamp_(eps, 1-eps), dim=0) * -1)

        # Backpropagate
        loss.backward()

        # Step the optimizer
        self.optimizer.step()

        return (loss.item(), mentions_found, total_mentions,
                corefs_found, total_corefs, corefs_chosen)

    def evaluate(self, val_corpus, eval_script='../src/eval/scorer.pl'):
        """ Evaluate a corpus of CoNLL-2012 gold files """

        # Predict files
        print('Evaluating on validation corpus...')
        predicted_docs = [self.predict(doc) for doc in tqdm(val_corpus)]
        val_corpus.docs = predicted_docs

        # Output results
        golds_file, preds_file = self.to_conll(val_corpus, eval_script)

        # Run perl script
        print('Running Perl evaluation script...')
        p = Popen([eval_script, 'all', golds_file, preds_file], stdout=PIPE)
        stdout, stderr = p.communicate()
        results = str(stdout).split('TOTALS')[-1]

        # Write the results out for later viewing
        with open('../preds/results.txt', 'w+') as f:
            f.write(results)
            f.write('\n\n\n')

        return results

    def predict(self, doc):
        """ Predict coreference clusters in a document """

        # Set to eval mode
        self.model.eval()

        # Initialize graph (mentions are nodes and edges indicate coref linkage)
        graph = nx.Graph()

        # Pass the document through the model
        spans, probs = self.model(doc)

        # Cluster found coreference links
        for i, span in enumerate(spans):

            # Loss implicitly pushes coref links above 0, rest below 0
            found_corefs = [idx
                            for idx, _ in enumerate(span.yi_idx)
                            if probs[i, idx] > probs[i, len(span.yi_idx)]]

            # If we have any
            if any(found_corefs):

                # Add edges between all spans in the cluster
                for coref_idx in found_corefs:
                    link = spans[coref_idx]
                    graph.add_edge((span.i1, span.i2), (link.i1, link.i2))

        # Extract clusters as nodes that share an edge
        clusters = list(nx.connected_components(graph))

        # Initialize token tags
        token_tags = [[] for _ in range(len(doc))]

        # Add in cluster ids for each cluster of corefs in place of token tag
        for idx, cluster in enumerate(clusters):
            for i1, i2 in cluster:
  
                if i1 == i2:
                    token_tags[i1].append(f'({idx})')

                else:
                    token_tags[i1].append(f'({idx}')
                    token_tags[i2].append(f'{idx})')

        doc.tags = ['|'.join(t) if t else '-' for t in token_tags]

        return doc


    def save_model(self, savepath):
        """ Save model state dictionary """
        torch.save(self.model.state_dict(), savepath + '.pth')

    def load_model(self, loadpath):
        """ Load state dictionary into model """
        state = torch.load(loadpath)
        self.model.load_state_dict(state)
        self.model = to_cuda(self.model)

