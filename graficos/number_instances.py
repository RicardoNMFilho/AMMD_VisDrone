from pathlib import Path
from collections import defaultdict

DATASETS = {
    "Original": Path("../datasets/VisDrone2019-CROPS"),
    "Redimensionado": Path("../datasets/VisDrone2019-CROPS-224"),
    "Com Augmentation": Path("../datasets/VisDrone2019-CROPS-224-AUGMENTATION"),
}

def contar_imagens_por_classe(dataset_path):
    contagem = defaultdict(lambda: defaultdict(int))  # split -> classe -> contagem
    for split_dir in dataset_path.iterdir():
        if not split_dir.is_dir():
            continue
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            total = len(list(class_dir.glob("*.jpg")))
            contagem[split_dir.name][class_dir.name] = total
    return contagem

# Conta para todos os datasets
contagens_datasets = {}
for nome_dataset, path in DATASETS.items():
    contagens_datasets[nome_dataset] = contar_imagens_por_classe(path)

# Coleta todos os splits e classes possíveis
splits = set()
classes = set()
for contagem in contagens_datasets.values():
    splits.update(contagem.keys())
    for split in contagem:
        classes.update(contagem[split].keys())

splits = sorted(splits)
classes = sorted(classes)

# Imprime resultado
print("\n====== COMPARAÇÃO DE INSTÂNCIAS POR CLASSE E DATASET ======\n")
for split in splits:
    print(f"--- SPLIT: {split} ---")
    header = f"{'Classe':<20}" + "".join([f"{nome:<20}" for nome in DATASETS.keys()])
    print(header)
    print("-" * len(header))
    for classe in classes:
        linha = f"{classe:<20}"
        for nome_dataset in DATASETS:
            contagem = contagens_datasets[nome_dataset]
            linha += f"{contagem.get(split, {}).get(classe, 0):<20}"
        print(linha)
    print()
