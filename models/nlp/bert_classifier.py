from transformers import BertForSequenceClassification, BertTokenizerFast
import torch.nn as nn

def load_bert_classifier(model_name='bert-base-uncased', num_labels=2):
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    return model, tokenizer
