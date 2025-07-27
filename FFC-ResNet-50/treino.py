import torch
import torch.nn as nn
import sys
import os

sys.path.append('FFC')

from model_zoo.ffc_resnet import ffc_resnet50

import random
import time
import numpy as np
import pandas as pd

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam

from datetime import datetime

def main():
    data_dir = '../datasets/VisDrone2019-CROPS-224'
    batch_size = 32
    num_epochs = 20
    lr = 0.0001
    patience = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Treinando em: {device}")
    if device.type == "cuda":
        print(f"🖥️  GPU detectada: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  GPU não detectada. Usando CPU.")

    nome_peso = f"pesos/ffc_resnet50_bs{batch_size}_ep{num_epochs}_lr{str(lr).replace('.', '')}_bal_{datetime.now().strftime('%Y%m%d-%H%M%S')}.pth"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_full = ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    val_ds = ImageFolder(os.path.join(data_dir, 'val'), transform=transform)

    num_classes = len(train_full.classes)
    print(f"\n📊 Número de classes: {num_classes}")

    # Balanceamento simples por undersampling
    targets = np.array(train_full.targets)
    min_count = np.bincount(targets).min()
    balanced_idxs = []
    for cls in range(num_classes):
        cls_idxs = np.where(targets == cls)[0]
        balanced_idxs.extend(random.sample(list(cls_idxs), min_count))
    random.shuffle(balanced_idxs)

    train_ds = Subset(train_full, balanced_idxs)

    print(f"📁 Instâncias no treino (balanceado): {len(train_ds)}")
    print(f"📁 Instâncias na validação: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    print(f"🌀 Batches por época (treino): {len(train_loader)}")
    print(f"🌀 Batches por época (validação): {len(val_loader)}\n")

    # --- MODELO FFC RESNET50 ---
    model = ffc_resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    checkpoint_path = 'pesos/FFC_ResNet_50.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Remover os pesos da última camada, que tem dimensões diferentes
    checkpoint.pop('fc.weight', None)
    checkpoint.pop('fc.bias', None)

    model.load_state_dict(checkpoint, strict=False)
    model = model.to(device)

    print("Modelo FFC ResNet50 carregado com pesos do ImageNet (exceto fc) e pronto para fine-tuning.")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    wait = 0
    logs = []

    for epoch in range(num_epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        t0 = time.time()

        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)

            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()

            _, pred = out.max(1)
            correct += (pred == lbls).sum().item()
            total += lbls.size(0)
            loss_sum += loss.item() * imgs.size(0)

        train_acc = 100 * correct / total
        train_loss = loss_sum / total

        model.eval()
        val_total, val_correct, val_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                loss = criterion(out, lbls)

                _, pred = out.max(1)
                val_correct += (pred == lbls).sum().item()
                val_total += lbls.size(0)
                val_loss_sum += loss.item() * imgs.size(0)

        val_acc = 100 * val_correct / val_total
        val_loss = val_loss_sum / val_total
        dur = (time.time() - t0) / 60

        logs.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "duration_min": dur
        })

        print(f"[{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}% | "
              f"Time: {dur:.2f} min")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wait = 0
            torch.save(model.state_dict(), nome_peso)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping ativado na época {epoch+1}")
                break

    pd.DataFrame(logs).to_csv("metricas_treino.csv", index=False)
    print(f"\n✅ Melhores pesos salvos em: {nome_peso}")


if __name__ == "__main__":
    main()