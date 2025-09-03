import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import torch
import utils
import model
from datetime import datetime as d

print("PyTorch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("cudnn version:", torch.backends.cudnn.version())
print("Number of GPUs:", torch.cuda.device_count())
print("Current cuda device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("")

device = "cuda" if torch.cuda.is_available() else "cpu"
block_size = 128  # AKA ctx_len
batch_size = 64
save_path = "./SmolGPT.pth"

print("Using " + device + " device.")

with open("./dataset/articles.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Lenght of the Dataset:", len(text))

tok = utils.Tokenizer(utils.Tokenizer.get_vocab(text))
vocab_size = len(tok.get_vocab(text))

print(f"Vocab size: {vocab_size}")

data = torch.tensor(tok.encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
test_data = data[n:]


def get_batch(split="train"):
    data = train_data if split == "train" else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)

    return x, y


xb, yb = get_batch()
print(f"inputs: {xb.shape}")
print(xb)
print(f"targets: {yb.shape}")
print(yb)
print("--------------")

model.model_config = model.ModelConfig(
    device, vocab_size, batch_size, block_size, 512, 8, 12, 0.2
)

model = model.BigramLanguageModel() # torch.load(save_path)
m = model.to(device)
out, loss = model(xb, yb)
print(out.shape)
print(loss)

optimizer = torch.optim.AdamW(m.parameters(), lr=4e-5)

losses = []
training_steps = 180000
log_step = 1000
try:
    for step in range(training_steps + 1):
        # Sample a batch of training data
        xb, yb = get_batch()

        # Actual training
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step % 1000 == 0:
            torch.save(model, save_path)
            time = d.now()
            time = time.strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"({time}) [{step/log_step}/{training_steps//log_step}] Loss: {sum(losses) / len(losses)}   Model saved."
            )
            losses = []
except KeyboardInterrupt:
    pass

prompt = "Emmanuel"
inputs = torch.tensor([tok.encode(prompt)], dtype=torch.long)
inputs = inputs.to(device)
print("inputs: ", inputs)
print(prompt, end="", flush=True)
for token in m.generate(inputs, 3200):
    print(tok.decode(token), end="", flush=True)

torch.save(model, save_path)
print("Model saved to " + save_path)
