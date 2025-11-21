# Machine Learning Projects - Zero to Mastery

Este repositorio contiene proyectos educativos de Machine Learning y Deep Learning siguiendo la metodología del curso **Zero to Mastery**. Cada proyecto implementa un flujo de trabajo completo de ML utilizando el **marco de 6 pasos**:

1. **Definición del Problema**
2. **Análisis de Datos**
3. **Métricas de Evaluación**
4. **Ingeniería de Características**
5. **Modelado**
6. **Experimentación**

## 🚀 Proyectos

### 1. 🚜 BlueBookForBulldozers - Predicción de Precios
**Tipo:** Regresión con Series Temporales

Predice el precio de venta de bulldozers utilizando datos históricos de subastas de Kaggle.

- **Dataset:** 400,000+ ejemplos con 50+ características
- **Métrica:** RMSLE (Root Mean Squared Log Error)
- **Desafíos:** Características temporales, valores faltantes, codificación categórica
- **Notebook:** `end-to-end-bluebook-bulldozer-price-regression.ipynb`

### 2. ❤️ HeartDiseaseProject - Predicción de Enfermedad Cardíaca
**Tipo:** Clasificación Binaria

Predice la presencia de enfermedad cardíaca basándose en parámetros clínicos del paciente.

- **Dataset:** 303 pacientes, 14 características (edad, sexo, presión arterial, colesterol, etc.)
- **Métrica:** Accuracy (objetivo: 95%)
- **Características:** Datos numéricos, sin valores faltantes, clases balanceadas
- **Notebook:** `end-to-end heart disease predictions.ipynb`

### 3. 🐶 DogClassifierDL - Clasificador de Razas de Perros
**Tipo:** Deep Learning - Clasificación Multiclase (120 clases)

Clasifica imágenes de perros en 120 razas diferentes usando Transfer Learning.

- **Dataset:** 20,000+ imágenes del Stanford Dogs Dataset
- **Enfoque:** Transfer Learning con CNNs preentrenadas de TensorFlow Hub
- **Métrica:** Accuracy (meta: superar 22% del paper original)
- **Entorno:** Optimizado para Google Colab con GPU
- **Notebook:** `end_to_end_dog_vision.ipynb`

## 📚 Stack Tecnológico

### Librerías Principales
- **pandas** - Manipulación y análisis de datos
- **NumPy** - Operaciones numéricas
- **matplotlib / seaborn** - Visualización de datos
- **scikit-learn** - Algoritmos de ML tradicionales
- **TensorFlow / Keras** - Deep Learning
- **TensorFlow Hub** - Transfer Learning con modelos preentrenados

### Notebooks Introductorios
- `Course.ipynb` - Introducción al curso
- `Numpy Introduction.ipynb` - Fundamentos de NumPy
- `Sci-kit-learn Introduction.ipynb` - Conceptos básicos de scikit-learn

## 🛠️ Configuración del Entorno

### Requisitos Previos
- Python 3.x
- Jupyter Notebook o JupyterLab
- Entorno virtual (Conda recomendado)

### Instalación

1. **Clonar el repositorio:**
```bash
git clone <repository-url>
cd project1ztm
```

2. **Activar el entorno virtual:**
```powershell
# Windows
.\env\Scripts\Activate.ps1

# O si usas conda
conda activate .\env
```

3. **Instalar dependencias principales:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
# Para deep learning:
pip install tensorflow tensorflow-hub
```

4. **Iniciar Jupyter:**
```bash
jupyter lab
# o
jupyter notebook
```

## 📊 Datasets

### Incluidos en data/
- `car-sales.csv` - Datos de ventas de autos (ejercicios de pandas)
- `car-sales-extended.csv` - Versión extendida
- `car-sales-extended-missing-data.csv` - Dataset con valores faltantes intencionales
- `heart-disease.csv` - Dataset de enfermedad cardíaca

### Descargar Separadamente
- **Bulldozer Prices:** [Kaggle - Bluebook for Bulldozers](https://www.kaggle.com/c/bluebook-for-bulldozers)
- **Dog Images:** [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)

## 📖 Flujo de Trabajo Común

Todos los proyectos siguen un patrón consistente:

1. Importar librerías necesarias
2. Cargar y explorar datos (EDA)
   - `df.info()`, `df.describe()`, `df.head()`
   - Verificar valores faltantes: `df.isna().sum()`
   - Distribución de clases: `df['target'].value_counts()`
3. Preparación de datos
   - Separar características y etiquetas
   - División train/test
   - Manejo de valores faltantes
   - Codificación de variables categóricas
4. Entrenamiento de modelos (probar múltiples algoritmos)
5. Evaluación con métricas apropiadas
6. Ajuste de hiperparámetros (GridSearchCV/RandomizedSearchCV)
7. Validación cruzada
8. Análisis de importancia de características
9. Guardar modelo final (.joblib para sklearn, SavedModel para TensorFlow)

## 🎯 Métricas de Evaluación

### Clasificación
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, Confusion Matrix

### Regresión
- MAE, MSE, RMSE, RMSLE
- R² (Coeficiente de determinación)

## 🔑 Conceptos Clave Demostrados

### Manejo de Datos Faltantes
- Relleno con media/mediana/moda
- Eliminación de filas/columnas
- Uso de imputadores de scikit-learn

### Codificación Categórica
- Label Encoding
- One-Hot Encoding (`pd.get_dummies()`)
- Ordinal Encoding

### Validación de Modelos
- Train/Test Split
- Cross-Validation (5-fold estándar)
- Comparación con baseline

## 📝 Notas Importantes

- Los notebooks están documentados en **español** con explicaciones detalladas
- El proyecto está diseñado con fines **educativos**, no para producción
- DogClassifierDL requiere recursos computacionales significativos (GPU recomendada)
- Los modelos guardados (.joblib, .pkl) no se deben versionar en git
- Los archivos comprimidos (.7z, .zip) están excluidos del control de versiones

## 📚 Recursos Adicionales

### Fuentes de Datos
- [UCI ML Repository - Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
- [Kaggle - Bluebook for Bulldozers](https://www.kaggle.com/c/bluebook-for-bulldozers)
- [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)

### Referencias del Curso
- Inspirado en [fast.ai ML course](https://course18.fast.ai/ml)
- Framework de 6 pasos para ML

## 🤝 Contribuciones

Este es un repositorio educativo personal. Los proyectos siguen el contenido del curso Zero to Mastery de Machine Learning.

## 📄 Licencia

Repositorio con fines educativos y de aprendizaje personal.

---

**Última actualización:** Noviembre 2025
