import torch

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def generate_adversarial(model, data, target, epsilon=0.1, device='cpu'):
    data, target = data.to(device), target.to(device)
    data.requires_grad = True
    output = model(data)
    loss = torch.nn.CrossEntropyLoss()(output, target)
    model.zero_grad()
    loss.backward()
    data_grad = data.grad.data
    perturbed_Data = fgsm_attack(data, epsilon, data_grad)
    return perturbed_Data