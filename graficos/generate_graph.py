from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

ROOT = Path("datasets/VisDrone2019-CROPS-224")

SPLITS = ["train", "val", "test-dev"]

class_counts = defaultdict(int)

for split in SPLITS:
    split_dir = ROOT / split
    if not split_dir.exists():
        continue
    for class_dir in split_dir.iterdir():
        if class_dir.is_dir():
            n_imgs = len(list(class_dir.glob("*.jpg")))
            class_counts[class_dir.name] += n_imgs

sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
classes, counts = zip(*sorted_counts)

plt.figure(figsize=(12, 6))
bars = plt.bar(classes, counts, color='skyblue', edgecolor='black')
plt.title("Número de instâncias por classe no VisDrone2019-CROPS-224")
plt.xlabel("Classe")
plt.ylabel("Número de imagens")
plt.xticks(rotation=45)
plt.tight_layout()

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 10, f'{yval:.0f}', ha='center', va='bottom')

plt.show()
