import torch
import torch.nn.functional as F
from src.uti import clamp_normalized

@torch.no_grad()
def _predict(model, x):
    return model(x).argmax(dim=1)

def fgsm_attack(model, x, y, epsilon, device='cpu', loss_fn=None):
    """
    x, y are already normazlied by the DataLoader transforms
    epsilon is in *normalized* units. use epsilon_norm = epsilon_pixel / 0.3081 for epsilon in pixel space.
    """
    model.eval()
    x, y = x.to(device), y.to(device)
    x_adv = x.clone().detach().requires_grad_(True)

    if loss_fn is None:
        # Use CE on logits (recommended for simpleCNN)
        loss_fn = torch.nn.CrossEntropyLoss()

    logits = model(x_adv)
    loss = loss_fn(logits, y)

    model.zero_grad(set_to_none=True)
    loss.backward()
    # FGSM step in normalized space
    x_adv = x_adv + epsilon * x_adv.grad.detach().sign()
    x_adv = clamp_normalized(x_adv).detach()
    return x_adv

def pgd_attack(model, x, y, epsilon, step_size, num_steps, device='cpu', loss_fn=None, random_start=True):
    """
    L-infinity PGD in normalized space.
    epsilon, step_size in normalized units
    """
    model.eval()
    x, y = x.to(device), y.to(device)
    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss()

    # Keep a copy of the clean input
    x_orig = x.detach()

    if random_start:
        # Start within the epsilon-ball
        delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
        x_adv = clamp_normalized(x_orig + delta)
    else:
        x_adv = x_orig.clone()
    
    x_adv.requires_grad_(True)

    for _ in range(num_steps):
        logits = model(x_adv)
        loss = loss_fn(logits, y)

        model.zero_grad(set_to_none=True)
        if x_adv.grad is not None:
            x_adv.grad.zero_()
        loss.backward()

        with torch.no_grad():
            # Gradient ascent on loss
            x_adv += step_size * x_adv.grad.sign()
            # Project back to epsilon-ball around x_orig
            x_adv = torch.max(torch.min(x_adv, x_orig + epsilon), x_orig - epsilon)
            x_adv = clamp_normalized(x_adv)
        x_adv.requires_grad_(True)

    return x_adv.detach()


def generate_adversarial(model, data, target, epsilon=0.1, device='cpu'):
    # Use the proper fgsm_attack function
    return fgsm_attack(model, data, target, epsilon, device=device)

def generate_pgd_adversarial(model, data, target, epsilon=0.1, alpha=0.01, steps=40, device='cpu'):
    # Use the proper pgd_attack function
    return pgd_attack(model, data, target, epsilon, alpha, steps, device=device)