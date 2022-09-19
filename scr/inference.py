from .model import TweetCatModel
import os
import yaml
from language_models.pretrainedbert.roberta import preprocess
from tqdm import tqdm
import torch

def load_checkpoint_model(path, model_name):
    path_params = os.path.join(path, "hparams.yaml")
    path_model = os.path.join(path, model_name)

    model = TweetCatModel.load_from_checkpoint(
        checkpoint_path=path_model, hparams_file=path_params, map_location=torch.device('cuda')
    )
    model.eval()
    model.cuda()
    return model

def chunks(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data[i: i+chunk_size]


def predict(data, model, tokenizer, chunk_size=64, return_confidence=False):
    len_data = len(data)
    data_chunks = chunks(list(data), chunk_size)
    res = []
    for chunk in tqdm(data_chunks, total=int(len_data / chunk_size)):
        date = torch.LongTensor([[ex["date"].day - 1, ex["date"].month - 1] for ex in chunk])
        days = date[:,0].to(device='cuda')
        months = date[:,1].to(device='cuda')
        texts = [ex["text"] for ex in chunk]
        texts = [preprocess(item) for item in texts]
        out_tokenizer = tokenizer(texts, truncation=True, padding=True, max_length=128)
        input_ids = torch.LongTensor(out_tokenizer['input_ids']).to(device='cuda')
        attention_mask = torch.LongTensor(out_tokenizer['attention_mask']).to(device='cuda')

        with torch.no_grad():
            scores = model(input_ids, days, months, attention_mask).squeeze(dim=1)#.logits[:, [0, 2]].argmax(dim=1)
        res += scores
    return res