from model import CorefModel
from gensim.models import KeyedVectors 
from trainer import Trainer

if __name__ == "__main__":
    
    w2v_model = KeyedVectors.load('embedding/Tencent_AILab_ChineseEmbedding.bin')
    model = CorefModel(embeds_dim=100,hidden_dim=50,embed_model=w2v_model)

    trainer = Trainer(model)
    
    trainer.train(10)