import torch
import matplotlib.pyplot as plt
from .uti import denormalize
from typing import Callable, List, Tuple

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

def test_adversarial(model, test_loader, attack_fn: Callable, attack_kwargs: dict, device='cpu'):
    """
    attack_fn signature must be (model, x, y, **attack_kwargs)
    """
    model.to(device)
    model.eval()
    correct = 0
    n = len(test_loader.dataset)
    examples = []

    for data, target in test_loader:
        data_dev, target_dev = data.to(device), target.to(device)
        adv = attack_fn(model, data_dev, target_dev, device=device, **attack_kwargs)
        with torch.no_grad():
            out = model(adv)
            pred = out.argmax(dim=1)
            correct += (pred == target_dev).sum().item()

        if len(examples) < 6:
            # Save CPU copies for easy plotting
            examples.append((
                data.cpu().detach(),
                adv.cpu().detach(),
                target.cpu().detach(),
                pred.cpu().detach()
            ))

    acc = correct / n
    return acc, examples
    
def visualize_adversarial(original, perturbed, label_original=None, label_perturbed=None):
    # original/perturbed assumed normalized; convert to [0,1] for display
    img_orig = denormalize(original[0][0].cpu().detach())
    img_adv = denormalize(perturbed[0][0].cpu().detach())

    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plt.imshow(img_orig, cmap='gray')
    plt.title(f'Original: {label_original}')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(img_adv, cmap='gray')
    plt.title(f"Perturbed: {label_perturbed}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def sweep_attack_accuracies(model, test_loader, make_attack: Callable[[float], Tuple[Callable, dict]], eps_list: List[float], device='cpu'):
    """
    make_attack(eps) -> (attack_fn, attack_kwargs)
    Returns list of (epsilon, accuracy)
    """
    results = []
    for eps in eps_list:
        attack_fn, attack_kwargs = make_attack(eps)
        acc, _ = test_adversarial(model, test_loader, attack_fn, attack_kwargs, device=device)
        print(f"epsilon={eps:.4f} -> adversarial accuracy = {acc*100:.2f}% ")
        results.append((eps, acc))
    return results

def plot_accuracy_vs_epsilon(results, title="Adversarial Robustness"):
    eps, acc = zip(*results)
    plt.figure(figsize=(5,4))
    plt.plot(eps, [a*100 for a in acc], marker='o')
    plt.xlabel("epsilon (normalized space)")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    
