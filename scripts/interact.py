from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class DialogueManager:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.history = []

    def update_state(self, user_input):
        self.history.append(user_input)

    def generate_response(self):
        input_text = " ".join(self.history)
        inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(inputs, max_length=500, pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
        return response

def interact():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('models')
    model.to(device)
    dialogue_manager = DialogueManager(model, tokenizer)

    print("Welcome to the movie recommender system!")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        dialogue_manager.update_state(user_input)
        response = dialogue_manager.generate_response()
        print(f"Bot: {response}")

if __name__ == "__main__":
    interact()
