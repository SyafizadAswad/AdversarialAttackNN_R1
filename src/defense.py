import torch
import torch.nn as nn

def adversarial_training(model, train_loader, epsilon=0.1, epochs=3, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            data.requires_grad = True
            # Generate adversarial examples
            output = model(data)
            loss = criterion(output, target)
            model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            perturbed_data = data + epsilon * data_grad.sign()
            perturbed_data = torch.clamp(perturbed_data, 0, 1)

            # Train on perturbed data
            optimizer.zero_grad()
            output_adv = model(perturbed_data)
            loss_adv = criterion(output_adv, target)
            loss_adv.backward()
            optimizer.step()
            running_loss += loss_adv.item()
            print(f"Epoch {epoch+1}/{epochs} - Adv Loss: {running_loss/len(train_loader):.4f}")
        return model