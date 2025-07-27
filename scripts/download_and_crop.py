import os
import zipfile
import cv2
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import requests

ROOT = Path("../datasets/VisDrone")
DEST_ROOT = Path("../datasets/VisDrone2019-CROPS")
SPLITS = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
    "test-dev": "VisDrone2019-DET-test-dev"
}
URLS = {
    "VisDrone2019-DET-train.zip": "https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-train.zip",
    "VisDrone2019-DET-val.zip": "https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-val.zip",
    "VisDrone2019-DET-test-dev.zip": "https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-test-dev.zip",
}

ID_TO_CLASS = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor"
}

SCORE_THRES = 0.0
MIN_AREA = 32 * 32
PADDING = 4

def download_file(url, dest_path):
    if dest_path.exists():
        print(f"[✓] {dest_path.name} já existe, pulando download.")
        return

    print(f"[↓] Baixando {url} ...")
    try:
        response = requests.get(url, stream=True)
        total = int(response.headers.get('content-length', 0))
        with open(dest_path, 'wb') as file, tqdm(
            desc=f"[→] {dest_path.name}",
            total=total,
            unit='B',
            unit_scale=True,
            unit_divisor=1024
        ) as bar:
            for data in response.iter_content(chunk_size=1024 * 1024):
                file.write(data)
                bar.update(len(data))
        print(f"[✓] Download finalizado: {dest_path}")
    except Exception as e:
        print(f"[✗] Erro ao baixar {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        raise

def unzip_and_cleanup(zip_path, extract_to):
    print(f"[⤓] Extraindo {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"[✓] Extraído para {extract_to}")
    zip_path.unlink()

def safe_crop(img, x, y, w, h, pad=0):
    H, W = img.shape[:2]
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(W, x + w + pad), min(H, y + h + pad)
    return img[y1:y2, x1:x2]

def crop_instances(split_key, src_dir, dest_dir):
    ann_dir = src_dir / "annotations"
    img_dir = src_dir / "images"
    class_counters = defaultdict(int)

    for ann_file in tqdm(sorted(ann_dir.glob("*.txt")), desc=f"[✂] Processando {split_key}"):
        img_file = img_dir / ann_file.with_suffix(".jpg").name
        if not img_file.exists():
            continue

        img = cv2.imread(str(img_file))
        if img is None:
            continue

        with open(ann_file, 'r') as f:
            for line in f:
                vals = [v.strip() for v in line.split(",")]
                if len(vals) < 6:
                    continue
                x, y, w, h = map(int, vals[:4])
                score = float(vals[4])
                cls_id = int(vals[5])

                if score < SCORE_THRES or w * h < MIN_AREA or cls_id not in ID_TO_CLASS:
                    continue

                crop = safe_crop(img, x, y, w, h, pad=PADDING)
                if crop.size == 0:
                    continue

                class_name = ID_TO_CLASS[cls_id]
                out_dir = dest_dir / split_key / class_name
                out_dir.mkdir(parents=True, exist_ok=True)

                class_counters[class_name] += 1
                out_path = out_dir / f"{ann_file.stem}_{class_counters[class_name]:06d}.jpg"
                cv2.imwrite(str(out_path), crop)

    print(f"\n[✔] Recortes gerados para {split_key}:")
    for cls, count in sorted(class_counters.items()):
        print(f"    {cls:15s}: {count}")

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    for filename, url in URLS.items():
        zip_path = ROOT / filename
        download_file(url, zip_path)
        unzip_and_cleanup(zip_path, ROOT)

    for split_key, split_dir in SPLITS.items():
        crop_instances(
            split_key=split_key,
            src_dir=ROOT / split_dir,
            dest_dir=DEST_ROOT
        )

if __name__ == "__main__":
    main()