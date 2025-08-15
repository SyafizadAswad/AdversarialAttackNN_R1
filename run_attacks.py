import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.uti import MNIST_MEAN, MNIST_STD
from src.model import SimpleCNN
from src.attack import fgsm_attack, pgd_attack
from src.evaluate import (
    test_accuracy, test_adversarial,
    sweep_attack_accuracies, plot_accuracy_vs_epsilon,
    visualize_adversarial
)


def get_mnist_loaders(batch_size=128, num_workers=2):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return test_loader

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)

    # Load or create model
    model = SimpleCNN().to(device)
    
    # Try to load pre-trained model, otherwise use untrained model
    try:
        model.load_state_dict(torch.load("checkpoints/mnist_cnn.pt", map_location=device))
        print("Loaded pre-trained model from checkpoints/mnist_cnn.pt")
        model.eval()
    except FileNotFoundError:
        print("No pre-trained model found. Using untrained model for demonstration.")
        print("Note: Results will be poor since the model is not trained.")
        model.eval()

    # Data
    test_loader = get_mnist_loaders(batch_size=256)

    # 1) Clean accuracy
    clean_ac = test_accuracy(model, test_loader, device=device)

    # 2) FGSM sweep
    fgsm_eps = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3] # normalized
    def make_fgsm(eps):
        return fgsm_attack, {"epsilon": eps}

    print("\n=== FGSM Sweep ===")
    fgsm_results = sweep_attack_accuracies(model, test_loader, make_fgsm, fgsm_eps, device=device)
    plot_accuracy_vs_epsilon(fgsm_results, title="FGSM: Accuracy vs Epsilon (normalized)")

    # 3) PGD sweep (stronger attack)    
    pgd_eps = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    def make_pgd(eps):
        return pgd_attack, {"epsilon": eps, "step_size": eps/4, "num_steps": 40, "random_start": True}
    
    print("\n=== PGD Sweep ===")
    pgd_results = sweep_attack_accuracies(model, test_loader, make_pgd, pgd_eps, device=device)
    plot_accuracy_vs_epsilon(pgd_results, title="PGD: Accuracy vs Epsilon (normalized)")

    # 4) Visualize a few adversarial examples for a single epsilon
    viz_eps = 0.15
    acc, examples = test_adversarial(model, test_loader, fgsm_attack, {"epsilon": viz_eps}, device=device)
    print(f"\nFGSM (eps={viz_eps}) adversarial accuracy: {acc*100:.2f}%")

    for i in range(min(3, len(examples))):
        orig, adv, y, y_pred = examples[i]
        visualize_adversarial(orig, adv, label_original=int(y[0]), label_perturbed=int(y_pred[0]))

if __name__ == "__main__":
    main()