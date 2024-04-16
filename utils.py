import torch
import torch.nn.functional as F
import json

def safe_div(x,y):
    if y == 0:
        return 0
    return x/y
    
def to_var(x):
    """ Convert a tensor to a backprop tensor and put on GPU """
    return to_cuda(x).requires_grad_()

def to_cuda(x):
    """ GPU-enable a tensor """
    # if torch.cuda.is_available():
    #     x = x.cuda()
    return x

def extract_gold_corefs(document):
    gold_labels = [0 for i in range(document["pronoun"]["indexFront"])]    
    
    for i in range(document["antecedentNum"]):
        if document[str(i)]["indexFront"] >= len(gold_labels):
            continue
        gold_labels[document[str(i)]["indexFront"]] = 1
        gold_labels[document[str(i)]["indexBehind"]] = 1
        
    return gold_labels

if __name__ == "__main__":
    with open("dataset/new_train/10.json","r") as f:
        data = json.load(f)
    gold_corefs, gold_mentions = extract_gold_corefs(data)

    for i in gold_corefs:
        print(i)

    print()
    for j in gold_mentions:
        print(j)