from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm

# ---------------- CONFIGURAÇÕES ----------------
SRC_DIR = Path("../datasets/VisDrone2019-CROPS")
DEST_DIR = Path("../datasets/VisDrone2019-CROPS-224")
TARGET_SIZE = (224, 224)
PADDING_COLOR = (0, 0, 0)
# ------------------------------------------------

def resize_with_padding(image: Image.Image, target_size=(224, 224), fill_color=(0, 0, 0)):
    old_size = image.size
    ratio = min(target_size[0] / old_size[0], target_size[1] / old_size[1])

    new_width  = int(old_size[0] * ratio)
    new_height = int(old_size[1] * ratio)
    new_size   = (new_width, new_height)

    image = image.resize(new_size, Image.Resampling.BILINEAR)

    new_img = Image.new("RGB", target_size, fill_color)
    offset = ((target_size[0] - new_size[0]) // 2,
              (target_size[1] - new_size[1]) // 2)
    new_img.paste(image, offset)
    return new_img


# ---------------- PROCESSAMENTO EM LOTE ----------------
for split_dir in SRC_DIR.iterdir():
    if not split_dir.is_dir():
        continue
    for class_dir in split_dir.iterdir():
        if not class_dir.is_dir():
            continue
        out_dir = DEST_DIR / split_dir.name / class_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(class_dir.glob("*.jpg"), desc=f"{split_dir.name}/{class_dir.name}"):
            try:
                img = Image.open(img_path).convert("RGB")
                resized = resize_with_padding(img, target_size=TARGET_SIZE, fill_color=PADDING_COLOR)
                resized.save(out_dir / img_path.name)
            except Exception as e:
                print(f"Erro ao processar {img_path}: {e}")
