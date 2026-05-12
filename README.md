# 🧠 β-VAE con Cyclical Annealing
### Aprendizaje de representaciones estructuradas en datos 2D y 3D

> **Asignatura:** Redes Neuronales y Aprendizaje Profundo  
> **Grado:** Ingeniería en Inteligencia Artificial — Universidad de Alicante (3º curso)  
> **Lenguaje:** ![Jupyter](https://img.shields.io/badge/Jupyter_Notebook-100%25-F37626?logo=jupyter&logoColor=white)
> ![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
> ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)

---

## 📌 Descripción

Este proyecto implementa un **Variational Autoencoder con parámetro β (β-VAE)** combinado con una estrategia de **Cyclical Annealing** para el entrenamiento del término KL de la función de pérdida. El objetivo es aprender representaciones latentes **disentangled** (desacopladas) sobre conjuntos de datos 2D y 3D.

El proyecto se divide en dos partes diferenciadas:

| Parte | Datos | Objetivo |
|-------|-------|----------|
| **Parte 1** | Datos 2D (imágenes) | β-VAE base + Cyclical Annealing |
| **Parte 2** | Datos 3D (nubes de puntos — ModelNet40) | Extensión a nubes de puntos 3D |

---

## 🗂️ Estructura del Repositorio

```
RNAP-Ciclycal-BetaVAE/
│
├── 📁 Parte 1/                  # β-VAE sobre datos 2D
│   └── (notebooks .ipynb)
│   └── MNIST 2D / 3D
│   └── CELEBA
│   └── FOOD101
│
├── 📁 Parte 2/                  # β-VAE sobre datos 3D
│   ├── (notebooks .ipynb)
│   └── modelnet40/
│   └── modelnet25/
│   └── modelnet10/
│   └── modelnet3/         
│
├── 📄 README.md
└── 📄 requeriments.txt          # Dependencias del proyecto
```

> 📦 Archivos completos (modelos, datos, resultados): [Google Drive](https://drive.google.com/drive/folders/1bTddEzXlOAoWenZbNqHRLk48cvdGTZLD?usp=sharing)

---

## ⚙️ Conceptos Clave

### β-VAE
Un VAE estándar minimiza:

$$\mathcal{L} = \mathbb{E}_{q}[\log p(x|z)] - \beta \cdot D_{KL}(q(z|x) \| p(z))$$

Al aumentar **β > 1**, se penaliza más fuertemente la divergencia KL, forzando un espacio latente más estructurado y **disentangled**.

### Cyclical Annealing
En lugar de aumentar β de forma monotónica, se emplea un esquema **cíclico**: el peso del término KL sube y baja periódicamente durante el entrenamiento. Esto evita el *posterior collapse* y permite explorar mejor el espacio latente.

```
β
1.0 ┤  /\    /\    /\
    │ /  \  /  \  /  \
0.0 ┼/    \/    \/    ─── épocas →
    ciclo 1  ciclo 2  ciclo 3
```

---

## 🚀 Instalación y Uso

### 1. Clonar el repositorio
```bash
git clone https://github.com/mfr73ua/RNAP-Ciclycal-BetaVAE.git
cd RNAP-Ciclycal-BetaVAE
```

### 2. Instalar dependencias
```bash
pip install -r requeriments.txt
```
---

## 📦 Dependencias Principales

| Librería | Versión | Uso |
|----------|---------|-----|
| `torch` | 2.11.0 | Framework de deep learning |
| `torchview` | 0.2.7 | Visualización de arquitecturas |
| `numpy` | 2.4.4 | Operaciones numéricas |
| `matplotlib` | 3.10.8 | Visualización de resultados |
| `scikit-learn` | 1.8.0 | Métricas y utilidades ML |
| `scipy` | 1.17.1 | Cálculo científico |
| `tqdm` | 4.67.3 | Barras de progreso |
| `bps` | (git) | Representación de nubes de puntos 3D |
| `xgboost` | 3.2.0 | Clasificación en espacio latente |

> Listado completo en [`requeriments.txt`](./requeriments.txt)

---

## 📊 Resultados Esperados

- **Parte 1:** Visualización del espacio latente 2D/3D, reconstrucciones, y comparativa con/sin Cyclical Annealing.
- **Parte 2:** Codificación de nubes de puntos 3D, interpolaciones en el espacio latente y evaluación de disentanglement.

---

## 🔗 Recursos

- 📁 [Google Drive con archivos completos](https://drive.google.com/drive/folders/1bTddEzXlOAoWenZbNqHRLk48cvdGTZLD?usp=sharing)
- 📄 Paper original β-VAE: *Higgins et al., 2017 — β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*
- 📄 Cyclical Annealing: *Fu et al., 2019 — Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing*

---

## 👥 Autores

Proyecto desarrollado por estudiantes de **3º de Ingeniería en IA** de la **Universidad de Alicante**.

---

*Universidad de Alicante · Grado en Ingeniería en Inteligencia Artificial · Curso 2025-26*
