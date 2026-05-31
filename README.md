# Segmentación de Vasos Retinianos para Detección de Retinopatía Diabética (U-Net)

Solución a la **Pregunta 2** del Examen Parcial de Redes Neuronales y Aprendizaje
Profundo. Implementación propia de **U-Net** en PyTorch para segmentación binaria
píxel a píxel de vasos en imágenes de fondo de ojo, con ablación, evaluación
cruzada **DRIVE → CHASE_DB1**, análisis cualitativo por grosor de vaso y
**CLAHE** como estrategia de adaptación de dominio.

El código se ejecuta de extremo a extremo con un único comando (`python main.py`).

---

> Examen Parcial - Maestría en Inteligencia Artificial · Curso de Redes Neuronales y Aprendizaje Profundo · Sección A · Grupo 7

Integrantes:
- Julio Machado Torres.
- Brigitte Scarlett Del Río Ricce.

Docente:
- Ph.D. Aldo Camargo Fernández Baca.

---

## Tabla de contenido

- [1. Requisitos](#1-requisitos)
- [2. Estructura del proyecto](#2-estructura-del-proyecto)
- [3. Entorno virtual en VS Code](#3-entorno-virtual-en-vs-code)
- [4. Descargar los datasets](#4-descargar-los-datasets)
- [5. Ejecución](#5-ejecución)
- [6. Análisis post-hoc por grosor de vaso](#6-análisis-post-hoc-por-grosor-de-vaso)
- [7. Salidas por corrida](#7-salidas-por-corrida)
- [8. Cómo cada entregable queda cubierto](#8-cómo-cada-entregable-queda-cubierto)
- [9. Solución de problemas](#9-solución-de-problemas)

---

## 1. Requisitos

- Python 3.11
- GPU NVIDIA con drivers CUDA (recomendado). Funciona en CPU pero es lento.
- VS Code con la extensión **Python** (de Microsoft).

Comprueba la GPU/driver:
```bash
nvidia-smi
```

---

## 2. Estructura del proyecto

```
retinal-vessel-segmentation/
├── main.py                  # Orquestador: ablación + cross-dataset + CLAHE
├── train.py                 # Entrena/evalúa UN modelo con todos los artefactos
├── experiments.py           # Bloques experimentales
├── config.py                # Configuración central / hiperparámetros
├── run_thickness_analysis.py  # ★ Script post-hoc: genera análisis por grosor de vaso
├── requirements.txt
├── environment.yml          # Alternativa con conda
├── scripts/
│   └── prepare_data.py      # Verificación de la estructura de datasets
├── src/
│   ├── data.py              # DRIVE / STARE / CHASE_DB1 + augmentations
│   ├── preprocessing.py     # CLAHE, canal verde, FOV automático
│   ├── models.py            # U-Net implementada desde cero (depth configurable)
│   ├── losses.py            # BCE, Dice y combinada, todas con FOV
│   ├── metrics.py           # Sens / Spec / F1 / AUC dentro del FOV + ROC + grids
│   ├── engine.py            # Bucles de entrenamiento/validación, AMP, early stop
│   ├── inference.py         # Test-time augmentation
│   └── failure_analysis.py  # Sensibilidad por grosor de vaso
├── .vscode/                 # Configs de ejecución/depuración listas para usar
├── data/                    # (lo creas tú) datasets descomprimidos
└── outputs/                 # (se genera) modelos, métricas, figuras
```

---

## 3. Entorno virtual en VS Code

Abre la carpeta en VS Code y, en la terminal integrada (``Ctrl+` ``):

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Instala PyTorch con CUDA correspondiente (ver `nvidia-smi`):

```bash
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU
pip install torch torchvision
```
> Comando actualizado para tu sistema: https://pytorch.org/get-started/locally/

Y el resto:
```bash
pip install -r requirements.txt
```

Selecciona el intérprete: `Ctrl+Shift+P` → **Python: Select Interpreter** → `.venv`.

Verifica GPU:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 4. Descargar los datasets

Los tres datasets son públicos pero **requieren descarga manual** (cada portal pide
registro o aceptación de licencia). Descarga, descomprime y organiza así:

```
data/
├── DRIVE/                       # https://drive.grand-challenge.org/
│   ├── training/
│   │   ├── images/      *.tif
│   │   ├── 1st_manual/  *_manual1.gif
│   │   └── mask/        *_mask.gif
│   └── test/
│       ├── images/      *.tif
│       └── mask/        *_mask.gif
│       # NOTA: la distribución oficial de DRIVE NO incluye 1st_manual/ en test/
│       # (las etiquetas estaban ocultas para el challenge). Si tu copia tampoco
│       # las trae, el código automáticamente hace una partición 3-vías sobre
│       # DRIVE-training para tener un test con GT.
├── STARE/                       # https://cecas.clemson.edu/~ahoover/stare/
│   ├── stare-images/    im####.ppm
│   ├── labels-ah/       im####.ah.ppm
│   └── labels-vk/       im####.vk.ppm
└── CHASEDB1/                    # https://blogs.kingston.ac.uk/retinal/chasedb1/
    ├── Image_##L.jpg / Image_##R.jpg
    ├── Image_##L_1stHO.png
    └── Image_##L_2ndHO.png
```

Verifica con:
```bash
python scripts/prepare_data.py
```
Esperado mínimo: `DRIVE training 20 con GT`, `CHASE_DB1 28 OK`. DRIVE-test con GT y
STARE son opcionales.

---

## 5. Ejecución

### Smoke test (≈3 épocas, ~minutos)
```bash
python main.py --quick
```

### Pipeline completo (un solo comando)
```bash
python main.py
```
Ejecuta y agrega:
- **Ablación** (pérdida BCE/Dice/combinada y profundidad 3 vs 4).
- **Generalización cruzada**: entrena en DRIVE, evalúa en DRIVE-test y CHASE_DB1.
- **Adaptación de dominio (CLAHE)**: compara baseline vs CLAHE y mide cuánto
  cierra la brecha entre DRIVE y CHASE.

Resumen en `outputs/SUMMARY.md` y `outputs/all_results.json`.

### Entrenar un solo modelo
```bash
python train.py --run-name baseline_drive
python train.py --loss bce --run-name bce_only
python train.py --depth 3 --run-name shallow_unet
python train.py --preprocess none --run-name no_clahe
```

### Por etapas
```bash
python main.py --stage ablation
python main.py --stage cross
python main.py --stage adaptation
```

### Desde VS Code
Pestaña **Run and Debug** (`Ctrl+Shift+D`) → elige una configuración → ▶.

---

## 6. Análisis post-hoc por grosor de vaso

**Importante:** las funciones de experimentos (`ablation_study`, `cross_dataset_study`
y `domain_adaptation_study`) llaman a `train_run(..., make_failure=False)` para no
agregar tiempo a cada corrida. Esto significa que **los archivos
`sensitivity_by_thickness.png` y `sensitivity_by_thickness.json` NO se generan
automáticamente** al correr `main.py`.

Para producirlos sobre un modelo ya entrenado, usa el script auxiliar
`run_thickness_analysis.py` que está en la raíz del proyecto. Carga el
`best_model.pt`, reconstruye el split exacto vía la semilla del `config.json`,
ejecuta la inferencia sobre el test set y aplica la transformada de distancia
euclidiana sobre las máscaras GT para calcular la sensibilidad por banda
(finos / medios / gruesos).

### Uso

```bash
# Con el venv activado, desde la raíz del proyecto
python run_thickness_analysis.py
python run_thickness_analysis.py --run-name C_clahe
python run_thickness_analysis.py --run-name C_baseline_noprep
python run_thickness_analysis.py --run-name shallow_unet --output-dir outputs
```

### Requisitos
Necesita que existan en `outputs/<run_name>/`:
- `config.json` — configuración con la que se entrenó (semilla, preprocesamiento, etc.).
- `best_model.pt` — pesos del mejor modelo guardados al entrenar.

Y que `data/DRIVE/` esté disponible (igual que cuando se entrenó).

### Salida
Crea los dos archivos en la misma `outputs/<run_name>/`:
- `sensitivity_by_thickness.png` — gráfica de barras por banda de grosor.
- `sensitivity_by_thickness.json` — tabla numérica (`sensitivity`, `n_pixels`,
  `thickness_range_px`) para cada banda.

Además imprime en consola un resumen comparando las bandas, útil para discutir
en el informe el cuello de botella arquitectónico sobre vasos finos.

> Tiempo aproximado: 30-60 segundos por run (no se entrena nada, solo una pasada
> de inferencia sobre el test split de DRIVE).

---

## 7. Salidas por corrida

Cada `run` deja en `outputs/<run_name>/`:

| Archivo | Contenido |
|---|---|
| `config.json` | Configuración exacta usada (reproducibilidad) |
| `data_info.json` | Tamaños de partición |
| `best_model.pt` | Pesos del mejor modelo (por F1 de validación) |
| `history.json` | Curvas de entrenamiento por época |
| `test_metrics.json` | Sensibilidad, especificidad, F1, AUC-ROC, accuracy |
| `roc_pixel.png` | Curva ROC píxel a píxel (todo el FOV) |
| `qualitative.png` | Imagen, GT, predicción y mapa FN/FP para 4 muestras |
| `sensitivity_by_thickness.png` / `.json` | Sensibilidad por banda de grosor de vaso (★ generados por `run_thickness_analysis.py`) |
| `eval_CHASEDB1/` | Métricas, ROC y ejemplos en el dataset cruzado |

Tablas agregadas en `outputs/`: `ablation.csv`, `cross_dataset.csv`,
`domain_adaptation.csv`.

---

## 8. Cómo cada entregable queda cubierto

1. **U-Net personalizada en PyTorch** → `src/models.py` (DoubleConv, Down, Up,
   UNet con profundidad configurable; sin usar `segmentation_models_pytorch`).
2. **Ablación sobre ≥2 factores** → `experiments.ablation_study`: función de
   pérdida (BCE / Dice / combinada) y profundidad (3 vs 4).
3. **Evaluación en DRIVE** → `train.py` reporta sens / spec / F1 / AUC-ROC.
4. **Generalización entre datasets** → `experiments.cross_dataset_study` entrena
   en DRIVE y evalúa en CHASE_DB1; reporta la brecha en `cross_dataset.csv`.
5. **Análisis cualitativo de fallos por grosor** → `src/failure_analysis.py`
   calcula sensibilidad para vasos finos / medios / gruesos a partir de la
   transformada de distancia del GT. El script auxiliar
   `run_thickness_analysis.py` (sección 6) lo ejecuta sobre cualquier modelo
   ya entrenado.
6. **Adaptación de dominio (CLAHE)** → `experiments.domain_adaptation_study`
   compara baseline vs CLAHE y cuantifica la reducción de la brecha.

---

## 9. Solución de problemas

- **`torch.cuda.is_available()` False** → reinstala torch con el `--index-url`
  CUDA correcto.
- **`CUDA out of memory`** → baja `--batch-size 2` o `--img-size 384`.
- **`No se encontró DRIVE`** → revisa que `data/DRIVE/training/images/*.tif` exista.
- **`num_workers` problema en Windows** → usa `--num-workers 0`.
- **`run_thickness_analysis.py` reporta `FileNotFoundError`** → confirma que la
  carpeta `outputs/<run_name>/` contiene tanto `config.json` como `best_model.pt`.
  Si entrenaste con `--quick` la corrida es válida; si solo corriste el pipeline
  completo (`main.py`), los runs disponibles son `A_*`, `B_*` y `C_*` según el
  estudio.
- **La banda "finos" aparece vacía en el JSON** → comportamiento esperado a 512×512
  con la transformada de distancia euclidiana discreta: el esqueleto del vaso
  cae en la banda "medios" (radio 1.0 px). El contraste entre medios y gruesos
  sigue siendo válido para discutir el cuello de botella arquitectónico.

---
