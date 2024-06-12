import json
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

def load_csv_data(filepath):
    return pd.read_csv(filepath)

def load_jsonl_data(filepath):
    with open(filepath, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def preprocess_dialog(dialog):
    for message in dialog['messages']:
        message['text'] = word_tokenize(message['text'].lower())
    return dialog

def preprocess_data(csv_filepath, jsonl_filepath, output_filepath):
    csv_data = load_csv_data(csv_filepath)
    dialogs = load_jsonl_data(jsonl_filepath)
    preprocessed_dialogs = [preprocess_dialog(dialog) for dialog in dialogs]
    with open(output_filepath, 'w') as f:
        json.dump(preprocessed_dialogs, f)

if __name__ == "__main__":
    preprocess_data('data/movies_with_mentions.csv', 'data/train_data.jsonl', 'data/preprocessed_data.json')
