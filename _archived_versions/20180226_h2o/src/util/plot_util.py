from matplotlib import pyplot as plt


def save_plot(filename: str, pad=1.25, tight=True, close=True, dpi=150):
    if tight:
        plt.tight_layout(pad=pad)

    plt.savefig(filename, dpi=dpi)

    if close:
        plt.close()
