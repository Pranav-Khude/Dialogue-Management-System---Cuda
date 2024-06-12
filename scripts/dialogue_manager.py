import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
from torch.utils.data import Dataset

class DialogDataset(Dataset):
    def __init__(self, dialogs, tokenizer):
        self.dialogs = dialogs
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        dialog = self.dialogs[idx]
        encoded_dialog = self.tokenizer(dialog, return_tensors='pt', padding=True, truncation=True)
        return {key: value.squeeze() for key, value in encoded_dialog.items()}

def load_preprocessed_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def extract_dialogs(dialogs):
    extracted_dialogs = []
    for dialog in dialogs:
        for turn in dialog['messages']:
            text = turn.get('text', None)
            if text:
                extracted_dialogs.append(text)
            else:
                print(f"Skipping empty or invalid text in dialog: {dialog}")
    return extracted_dialogs

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Set pad_token to eos_token
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.to(device)

    dialogs = load_preprocessed_data('data/preprocessed_data.json')
    extracted_dialogs = extract_dialogs(dialogs)

    dataset = DialogDataset(extracted_dialogs, tokenizer)

    # Use DataCollatorForLanguageModeling for GPT-2
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=1,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,  # Set the data collator
    )

    trainer.train()

if __name__ == "__main__":
    main()
