import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from src.uti import MNIST_MEAN, MNIST_STD
from src.model import SimpleCNN
from src.attack import fgsm_attack, pgd_attack
from src.evaluate import test_accuracy, test_adversarial, visualize_adversarial

# AutoAttack import
from autoattack import AutoAttack

# -------------------- Data Loader --------------------
def get_mnist_loaders(batch_size=128, num_workers=2):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return test_loader

# -------------------- Sweeps --------------------
def sweep_attack_accuracies(model, test_loader, attack_fn, epsilons, device, attack_kwargs={}):
    """Compute accuracy of model under adversarial attack for each epsilon."""
    results = []
    for eps in epsilons:
        correct = 0
        total = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # Construct kwargs for this attack
            kwargs = attack_kwargs.copy()
            kwargs['epsilon'] = eps
            adv_data = attack_fn(model, data, target, device=device, **kwargs)
            output = model(adv_data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
        acc = correct / total
        print(f"epsilon={eps:.4f} -> adversarial accuracy = {acc*100:.2f}%")
        results.append((eps, acc))
    return results

def plot_accuracy_vs_epsilon(results, title="Accuracy vs Epsilon"):
    epsilons = [x[0] for x in results]
    accuracies = [x[1]*100 for x in results]
    plt.figure(figsize=(6,4))
    plt.plot(epsilons, accuracies, marker='o')
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.grid(True)
    plt.show()

# -------------------- Visualization --------------------
def visualize_clean_fgsm_pgd(model, data, target, epsilon=0.15, device='cpu'):
    model.eval()
    data, target = data.to(device), target.to(device)

    fgsm_adv = fgsm_attack(model, data, target, epsilon=epsilon, device=device)
    pgd_adv = pgd_attack(model, data, target, epsilon=epsilon, step_size=epsilon/4, num_steps=40, device=device)

    for i in range(min(3, data.size(0))):
        plt.figure(figsize=(9,3))

        # Clean
        plt.subplot(1,3,1)
        plt.imshow(data[i][0].cpu().detach(), cmap='gray')
        plt.title(f"Clean: {target[i].item()}")

        # FGSM
        plt.subplot(1,3,2)
        plt.imshow(fgsm_adv[i][0].cpu().detach(), cmap='gray')
        pred = model(fgsm_adv[i].unsqueeze(0)).argmax(dim=1).item()
        plt.title(f"FGSM: {pred}")

        # PGD
        plt.subplot(1,3,3)
        plt.imshow(pgd_adv[i][0].cpu().detach(), cmap='gray')
        pred = model(pgd_adv[i].unsqueeze(0)).argmax(dim=1).item()
        plt.title(f"PGD: {pred}")

# -------------------- Multi-Norm AutoAttack --------------------
def evaluate_multi_norm(model, test_loader, device='cpu'):
    model.eval()
    # Collect all test images and labels
    X, Y = [], []
    for data, target in test_loader:
        X.append(data)
        Y.append(target)
    X = torch.cat(X, dim=0).to(device)
    Y = torch.cat(Y, dim=0).to(device)

    # Linf evaluation
    adversary = AutoAttack(model, norm='Linf', eps=0.3, version='standard', device=device)
    print("\n=== AutoAttack Linf Evaluation ===")
    x_adv_linf = adversary.run_standard_evaluation(X, Y, bs=128)

    # L2 evaluation
    adversary = AutoAttack(model, norm='L2', eps=2.0, version='standard', device=device)
    print("\n=== AutoAttack L2 Evaluation ===")
    x_adv_l2 = adversary.run_standard_evaluation(X, Y, bs=128)

    return X, Y, x_adv_linf, x_adv_l2

def visualize_multi_norm_examples(model, X, Y, x_adv_linf, x_adv_l2, num_examples=3):
    model.eval()
    for i in range(num_examples):
        plt.figure(figsize=(9,3))

        # Clean
        plt.subplot(1,3,1)
        plt.imshow(X[i][0].cpu().detach(), cmap='gray')
        plt.title(f"Clean: {Y[i].item()}")

        # Linf
        plt.subplot(1,3,2)
        plt.imshow(x_adv_linf[i][0].cpu().detach(), cmap='gray')
        pred = model(x_adv_linf[i].unsqueeze(0)).argmax(dim=1).item()
        plt.title(f"Linf: {pred}")

        # L2
        plt.subplot(1,3,3)
        plt.imshow(x_adv_l2[i][0].cpu().detach(), cmap='gray')
        pred = model(x_adv_l2[i].unsqueeze(0)).argmax(dim=1).item()
        plt.title(f"L2: {pred}")

# -------------------- Main --------------------
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)

    # Load model
    model = SimpleCNN().to(device)
    try:
        model.load_state_dict(torch.load("checkpoints/mnist_cnn_adv.pt", map_location=device))
        print("Loaded adversarially trained model from checkpoints/mnist_cnn_adv.pt")
    except FileNotFoundError:
        model.load_state_dict(torch.load("checkpoints/mnist_cnn.pt", map_location=device))
        print("Loaded pre-trained model from checkpoints/mnist_cnn.pt")
    model.eval()

    # Load data
    test_loader = get_mnist_loaders(batch_size=256)

    # 1) Clean Accuracy
    clean_acc = test_accuracy(model, test_loader, device=device)

    # 2) FGSM Sweep
    fgsm_eps = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    print("\n=== FGSM Sweep ===")
    fgsm_results = sweep_attack_accuracies(model, test_loader, fgsm_attack, fgsm_eps, device=device)
    plot_accuracy_vs_epsilon(fgsm_results, title="FGSM: Accuracy vs Epsilon")

    # 3) PGD Sweep
    pgd_eps = fgsm_eps
    print("\n=== PGD Sweep ===")
    pgd_results = sweep_attack_accuracies(model, test_loader, pgd_attack, pgd_eps, device=device, attack_kwargs={'step_size':0.05, 'num_steps':40})
    plot_accuracy_vs_epsilon(pgd_results, title="PGD: Accuracy vs Epsilon")

    # 4) Visualize a few adversarial examples
    viz_eps = 0.15
    acc, examples = test_adversarial(model, test_loader, fgsm_attack, {"epsilon": viz_eps}, device=device)
    print(f"\nFGSM (eps={viz_eps}) adversarial accuracy: {acc*100:.2f}%")
    for i in range(min(3, len(examples))):
        orig, adv, y, y_pred = examples[i]
        visualize_adversarial(orig, adv, label_original=int(y[0]), label_perturbed=int(y_pred[0]))

    # 5) Side-by-side Clean vs FGSM vs PGD (first batch)
    data_iter = iter(test_loader)
    data_batch, target_batch = next(data_iter)
    visualize_clean_fgsm_pgd(model, data_batch, target_batch, epsilon=viz_eps, device=device)

    # 6) Multi-norm AutoAttack
    X, Y, x_adv_linf, x_adv_l2 = evaluate_multi_norm(model, test_loader, device=device)
    visualize_multi_norm_examples(model, X, Y, x_adv_linf, x_adv_l2, num_examples=3)


if __name__ == "__main__":
    main()
