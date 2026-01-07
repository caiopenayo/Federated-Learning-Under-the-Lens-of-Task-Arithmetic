import matplotlib.pyplot as plt

def plot_test_curves(histories, title_suffix=""):
    plt.figure()
    for name, h in histories.items():
        plt.plot(h["test_loss"], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Test Loss")
    plt.legend()
    plt.title(f"Test Loss {title_suffix}")
    plt.show()

    plt.figure()
    for name, h in histories.items():
        plt.plot(h["test_acc"], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.title(f"Test Accuracy {title_suffix}")
    plt.show()
