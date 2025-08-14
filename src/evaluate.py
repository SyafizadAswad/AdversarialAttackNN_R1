import torch
import matplotlib.pyplot as plt

def test_accuracy(model, test_loader, device='cpu'):
    model.to(device)
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
    accuracy = correct / len(test_loader.dataset)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    return accuracy
    
def visualize_adversarial(original, perturbed, label_original=None, label_perturbed=None):
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plt.imshow(original[0][0].cpu().detach(), cmap='gray')
    plt.title(f"Original: {label_original}")
    plt.subplot(1,2,2)
    plt.imshow(perturbed[0][0].cpu().detach(), cmap='gray')
    plt.title(f"Perturbed: {label_perturbed}")
    plt.show()
    
