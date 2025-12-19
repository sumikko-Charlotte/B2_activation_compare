import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss

def train_one(activation, X_train, y_train, X_val, y_val, epochs=200, seed=42):
    clf = MLPClassifier(
        hidden_layer_sizes=(32, 32),
        activation=activation,
        solver="adam",
        learning_rate_init=0.01,
        max_iter=1,
        warm_start=True,
        random_state=seed
    )

    train_losses = []
    val_losses = []

    for _ in range(epochs):
        clf.fit(X_train, y_train)
        train_losses.append(float(clf.loss_))
        proba = clf.predict_proba(X_val)
        val_losses.append(float(log_loss(y_val, proba)))

    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    return acc, train_losses, val_losses

def main():
    np.random.seed(42)
    os.makedirs("results", exist_ok=True)

    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    activations = [
        ("ReLU", "relu"),
        ("Sigmoid", "logistic"),
        ("Tanh", "tanh"),
        ("Identity", "identity"),
    ]

    summary_lines = []

    plt.figure()
    for name, act in activations:
        print(f"Training with activation: {name}")
        acc, train_loss, val_loss = train_one(act, X_train, y_train, X_val, y_val)
        summary_lines.append(
            f"{name}: val_acc={acc:.4f}, final_train_loss={train_loss[-1]:.4f}, final_val_loss={val_loss[-1]:.4f}"
        )
        plt.plot(val_loss, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Log Loss")
    plt.title("MLP Activation Function Comparison (sklearn)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/loss_curves.png", dpi=200)

    with open("results/summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print("Saved results/loss_curves.png")
    print("Saved results/summary.txt")
    print("\n".join(summary_lines))

if __name__ == "__main__":
    main()
