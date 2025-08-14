import torch
from src.dataset import get_dataloaders
from src.model import SimpleCNN
from src.train import train
from src.attack import generate_adversarial
from src.evaluate import test_accuracy, visualize_adversarial
from src.defense import adversarial_training

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load data
train_loader, test_loader = get_dataloaders(batch_size=64)

# Initialize model
model = SimpleCNN()

# Train baseline
model = train(model, train_loader, epochs=3, device=device)

# Evaluate baseline
test_accuracy(model, test_loader, device=device)

# Generate one batch of advesarial examples
data_iter = iter(test_loader)
data, target = next(data_iter)
perturbed_data = generate_adversarial(model, data, target, epsilon=0.1, device=device)

# Evaluate on perturbed data
model.eval()
output = model(perturbed_data.to(device))
pred_labels = output.argmax(dim=1)
visualize_adversarial(data, perturbed_data, label_original=target[0].item(), label_perturbed=pred_labels[0].item())

# Optional: adversarial training
model_adv = adversarial_training(model, train_loader, epsilon=0.1, epochs=2, device=device)
test_accuracy(model_adv, test_loader, device=device)