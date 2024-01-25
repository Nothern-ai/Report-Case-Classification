# dataset.py
import pandas as pd
import json
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import re

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')


def clean_text(text):
    """
    :param text: origin text。
    :return: cleaned text。
    """
    if text.startswith('工单来源：'):
        text = re.sub(r"工单来源：.*?={2,}", "", text, flags=re.DOTALL)
        text = text.lstrip('\n').replace('\n留言', '')
        text = re.sub(r"-.*$", "", text, flags=re.DOTALL)
    text = re.sub(r"工单流转.*$", "", text, flags=re.DOTALL)
    return text


class PredictionDataset(Dataset):
    def __init__(self, tokenizer, data):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        inputs = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt', max_length=512,
                                       truncation=True)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}  # 移除批处理维度
        return inputs


def predict_on_dataset(model, file_path, label_map, device):
    data = pd.read_excel(file_path)
    data['来电内容'] = data['来电内容'].apply(clean_text)
    texts = data['来电内容'].tolist()

    dataset = PredictionDataset(tokenizer, texts)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    index_to_label = {v: k for k, v in label_map.items()}

    predictions = []
    model.eval()
    with torch.no_grad():
        for inputs in tqdm(dataloader):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            _, pred = torch.max(outputs.logits, dim=1)
            predictions.append(index_to_label[pred.item()])

    data['predicted_label'] = predictions
    data.to_excel('predicted_dataset.xlsx')


def predict_single_text(model, text, label_map, device):
    text = clean_text(text)
    print("Done Cleaned Text")
    inputs = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt', max_length=512, truncation=True)
    print("Done Tokenized")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    index_to_label = {v: k for k, v in label_map.items()}
    print("Next to predict")
    outputs = model(**inputs)
    print("Finish inference")
    _, pred = torch.max(outputs.logits, dim=1)
    print("The answer is:")
    return index_to_label[pred.item()]
