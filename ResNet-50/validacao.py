import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from thop import profile

import time
import os


def main():
    data_dir = '../datasets/VisDrone2019-CROPS-224'
    modelo_peso = 'pesos/resnet50_bs32_ep20_lr00001_bal_20250727-072909.pth'
    batch_size = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_ds = ImageFolder(os.path.join(data_dir, 'test-dev'), transform=transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    class_names = val_ds.classes
    num_classes = len(class_names)

    model = resnet50()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(modelo_peso, weights_only=False))
    model = model.to(device)
    model.eval()

    total, correct = 0, 0
    all_preds, all_labels = [], []
    t0 = time.time()

    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            _, preds = out.max(1)
            total += lbls.size(0)
            correct += (preds == lbls).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())

    inference_time = time.time() - t0
    val_acc = 100 * correct / total
    img_per_sec = total / inference_time

    print(f"Acurácia na Validação: {val_acc:.2f}%")
    print(f"Tempo de inferência: {inference_time:.2f}s")
    print(f"Imagens por segundo: {img_per_sec:.2f} img/s")

    # Matriz de confusão
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("matriz_confusao_val.pdf", format='pdf')
    plt.close()

    # Métricas por classe
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=range(num_classes))

    acuracia_classe = cm.diagonal() / cm.sum(axis=1)

    metricas_df = pd.DataFrame({
        "Classe": class_names,
        "Acuracia": acuracia_classe,
        "Precisao": precision,
        "Recall": recall,
        "F1-score": f1
    })
    metricas_df.to_csv("metricas_por_classe.csv", index=False)

    # Métricas globais (micro e macro)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='micro')
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro')

    # FLOPs e parâmetros
    dummy = torch.randn(1, 3, 224, 224).to(device)
    flops, params = profile(model, inputs=(dummy,), verbose=False)

    # Salvar métricas gerais
    with open("metricas_gerais.csv", "w") as f:
        f.write("Acuracia,Tempo_inferencia_seg,Imgs_por_seg,FLOPs,Parametros,"
                "Precision_micro,Recall_micro,F1_micro,"
                "Precision_macro,Recall_macro,F1_macro\n")
        f.write(f"{val_acc:.2f},{inference_time:.2f},{img_per_sec:.2f},{flops},{params},"
                f"{precision_micro:.4f},{recall_micro:.4f},{f1_micro:.4f},"
                f"{precision_macro:.4f},{recall_macro:.4f},{f1_macro:.4f}\n")

    print("Métricas salvas.")

if __name__ == "__main__":
    main()
