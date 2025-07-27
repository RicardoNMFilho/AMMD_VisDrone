from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import random

# ---------------- CONFIGURAÇÕES ----------------
SRC_DIR = Path("../datasets/VisDrone2019-CROPS")
DEST_DIR = Path("../datasets/VisDrone2019-CROPS-224-AUGMENTATION")
TARGET_SIZE = (224, 224)
PADDING_COLOR = (0, 0, 0)  # cor de fundo: preto
# ------------------------------------------------

def resize_with_padding(image: Image.Image, target_size=(224, 224), fill_color=(0, 0, 0)):
    old_size = image.size
    ratio = min(target_size[0] / old_size[0], target_size[1] / old_size[1])

    new_width = int(old_size[0] * ratio)
    new_height = int(old_size[1] * ratio)
    new_size = (new_width, new_height)

    image = image.resize(new_size, Image.Resampling.BILINEAR)

    new_img = Image.new("RGB", target_size, fill_color)
    offset = ((target_size[0] - new_size[0]) // 2,
              (target_size[1] - new_size[1]) // 2)
    new_img.paste(image, offset)
    return new_img

# ---------------- CONTAGEM DE IMAGENS POR CLASSE ----------------
split_class_counts = defaultdict(dict)

for split_dir in SRC_DIR.iterdir():
    if not split_dir.is_dir():
        continue
    for class_dir in split_dir.iterdir():
        if not class_dir.is_dir():
            continue
        count = len(list(class_dir.glob("*.jpg")))
        split_class_counts[split_dir.name][class_dir.name] = count

# ---------------- IDENTIFICA QUANTIDADE MÍNIMA ALVO ----------------
min_counts = {}
for split, class_counts in split_class_counts.items():
    counts = list(class_counts.values())
    min_count = min(counts)
    minority_class = min(class_counts, key=class_counts.get)
    # Supondo que a menor classe será duplicada (imagem + 2 aug = 3x)
    total_after_aug = split_class_counts[split][minority_class] * 3
    min_counts[split] = total_after_aug

# ---------------- AUGMENTATION ----------------
def augment_image(image: Image.Image):
    transforms = [
        lambda x: ImageOps.mirror(x),
        lambda x: ImageOps.flip(x),
        lambda x: x.rotate(90),
        lambda x: x.rotate(180),
        lambda x: x.rotate(270)
    ]
    return [t(image) for t in transforms]

for split_dir in SRC_DIR.iterdir():
    if not split_dir.is_dir():
        continue

    split = split_dir.name
    target_count = min_counts[split]

    for class_dir in split_dir.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        out_dir = DEST_DIR / split / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        img_paths = list(class_dir.glob("*.jpg"))
        resized_images = []

        for img_path in tqdm(img_paths, desc=f"{split}/{class_name}"):
            try:
                img = Image.open(img_path).convert("RGB")
                resized = resize_with_padding(img, target_size=TARGET_SIZE, fill_color=PADDING_COLOR)

                # salvar original
                out_name = out_dir / img_path.name
                resized.save(out_name)
                resized_images.append((img_path.stem, resized))
            except Exception as e:
                print(f"Erro ao processar {img_path}: {e}")

        # augmenta todas as imagens da menor classe (x3) e
        # as demais até atingir o total da menor classe
        current_count = len(resized_images)
        extra_needed = target_count - current_count

        idx = 0
        while extra_needed > 0:
            name, base_img = resized_images[idx % current_count]
            aug_imgs = augment_image(base_img)

            for i, aug in enumerate(aug_imgs):
                if extra_needed <= 0:
                    break
                aug.save(out_dir / f"{name}_aug{i}.jpg")
                extra_needed -= 1

            idx += 1
