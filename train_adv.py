import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.model import SimpleCNN
from src.attack import pgd_attack
from src.uti import clamp_normalized

# --- Settings ---
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 64
epochs = 5
epsilon = 0.3
step_size = epsilon / 4
num_steps = 40
learning_rate = 1e-3

# --- Data ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --- Model ---
model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# --- Adversarial Training Loop ---
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        # Generate adversarial examples
        data_adv = pgd_attack(model, data, target, epsilon=epsilon, step_size=step_size, num_steps=num_steps, device=device)

        # Forward + backward on adversarial examples
        optimizer.zero_grad()
        output = model(data_adv)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}," 
          f"Accuracy on adv examples: {acc*100:.2f}%")
    
    # --- Save model ---
    torch.save(model.state_dict(), 'checkpoints/mnist_cnn_adv.pt')
    print("Adversarially trained model saved to checkpoints/mnist_cnn_adv.pt")
    