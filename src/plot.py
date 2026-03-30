import matplotlib.pyplot as plt

def plot_loss(train_losses, val_losses, model_name):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Loss Curve")
    plt.legend()
    plt.savefig(f"{model_name}_loss.png")
    plt.close()


def plot_final_results(results):
    models = list(results.keys())
    perplexities = [results[m]["perplexity"] for m in models]
    accuracies = [results[m]["accuracy"] for m in models]

    # Perplexity plot
    plt.figure()
    plt.bar(models, perplexities)
    plt.xlabel("Models")
    plt.ylabel("Perplexity")
    plt.title("Model Comparison (Perplexity)")
    plt.savefig("perplexity_comparison.png")
    plt.close()

    # Accuracy plot
    plt.figure()
    plt.bar(models, accuracies)
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.title("Model Comparison (Accuracy)")
    plt.savefig("accuracy_comparison.png")
    plt.close()