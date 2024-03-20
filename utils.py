import torch
import torch.nn.functional as F
import json

def pad_and_stack(tensors, pad_size=None, value=0):
    """ Pad and stack an uneven tensor of token lookup ids.
    Assumes num_sents in first dimension (batch_first=True)"""

    # Get their original sizes (measured in number of tokens)
    sizes = [s.shape[0] for s in tensors]

    # Pad size will be the max of the sizes
    if not pad_size:
        pad_size = max(sizes)

    # Pad all sentences to the max observed size
    # TODO: why does pad_sequence blow up backprop time? (copy vs. slice issue)
    padded = torch.stack([F.pad(input=sent[:pad_size],
                                pad=(0, 0, 0, max(0, pad_size-size)),
                                value=value)
                          for sent, size in zip(tensors, sizes)], dim=0)

    return padded, sizes

def to_var(x):
    """ Convert a tensor to a backprop tensor and put on GPU """
    return to_cuda(x).requires_grad_()

def to_cuda(x):
    """ GPU-enable a tensor """
    # if torch.cuda.is_available():
    #     x = x.cuda()
    return x

def extract_gold_corefs(document):
    gold_corefs = []
    gold_mentions = []

    for span in document["spans"]:
        cur_span = (span["span"]["begin"], span["span"]["end"])
        gold_mentions.append(cur_span)
        for pre_span in span["pre_spans"]:
            gold_corefs.append((cur_span,(pre_span["begin"],pre_span["end"])))
    
    return gold_corefs, gold_mentions

if __name__ == "__main__":
    with open("dataset/new_train/10.json","r") as f:
        data = json.load(f)
    gold_corefs, gold_mentions = extract_gold_corefs(data)

    for i in gold_corefs:
        print(i)

    print()
    for j in gold_mentions:
        print(j)