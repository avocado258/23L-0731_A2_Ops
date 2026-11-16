import matplotlib.pyplot as plt
from pipeline import train_ds
for imgs, labels in train_ds.take(1):
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    for ax, img in zip(axes.flat, imgs):
        ax.imshow(img.numpy().astype("uint8"))
        ax.axis("off")
    plt.savefig("figures/augmented_batch.png")