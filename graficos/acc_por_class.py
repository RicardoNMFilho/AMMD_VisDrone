import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Carrega o CSV
df = pd.read_csv("../ResNet-50/resultados/AjusteCompleto/metricas_por_classe.csv")

# Define as métricas e posições para o gráfico
classes = df["Classe"]
metricas = ["Acuracia", "Precisao", "Recall", "F1-score"]
x = np.arange(len(classes))  # posição das barras no eixo X
largura = 0.2  # largura de cada barra

# Cria o gráfico
fig, ax = plt.subplots(figsize=(12, 6))

for i, metrica in enumerate(metricas):
    ax.bar(x + i * largura, df[metrica], width=largura, label=metrica)

# Configurações do gráfico
ax.set_ylabel("Pontuação")
ax.set_xlabel("Classe")
ax.set_title("Métricas por Classe")
ax.set_xticks(x + largura * (len(metricas) - 1) / 2)
ax.set_xticklabels(classes, rotation=45, ha='right')
ax.set_ylim(0, 1.1)
ax.legend()

plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Salva em PDF
plt.savefig("../ResNet-50/resultados/AjusteCompleto/metricas_por_classe.pdf", format='pdf')

# Opcional: fecha a figura para liberar memória
plt.close()
