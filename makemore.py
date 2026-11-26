from pathlib import Path
import matplotlib.pyplot as plt
import torch


def main() -> None:
    # Load names dataset (place names.txt in the same folder or update the path)
    names_path = Path("names.txt")
    words = names_path.read_text(encoding="utf-8").splitlines()

    print(f"Total words: {len(words)}")
    print(f"Min word length: {min(len(w) for w in words)}")
    print(f"Max word length: {max(len(w) for w in words)}")

    # Build character <-> index mappings ('.' is index 0, 'a'-'z' are 1-26)
    chars = sorted(set("".join(words)))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi["."] = 0
    itos = {i: s for s, i in stoi.items()}

    # Count bigrams in a 27x27 tensor
    N = torch.zeros((27, 27), dtype=torch.int32)
    for w in words:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            N[ix1, ix2] += 1

    # Visualize the bigram count matrix
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.imshow(N, cmap="Blues")

    for i in range(27):
        for j in range(27):
            chstr = itos[i] + itos[j]
            ax.text(j, i, chstr, ha="center", va="bottom", color="gray")
            ax.text(j, i, str(N[i, j].item()), ha="center", va="top", color="gray")

    ax.axis("off")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()