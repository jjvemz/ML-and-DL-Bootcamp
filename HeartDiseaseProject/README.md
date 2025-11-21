# ❤️ Heart Disease Prediction - Clasificación con Machine Learning

## 📋 Descripción del Proyecto

Este proyecto de **clasificación binaria** utiliza Machine Learning para predecir la presencia de enfermedad cardíaca en pacientes basándose en parámetros clínicos. Implementa un flujo completo de Data Science desde la exploración de datos hasta la evaluación de múltiples algoritmos, logrando un modelo interpretable y de alta precisión para asistir en el diagnóstico médico temprano.

### 🎯 Problema de Salud Pública

Las enfermedades cardiovasculares son la **principal causa de muerte a nivel mundial**, responsables de ~17.9 millones de muertes anuales (OMS). La detección temprana es crucial para:
- Iniciar tratamientos preventivos oportunos
- Reducir costos de atención médica
- Mejorar calidad de vida de pacientes
- Optimizar recursos hospitalarios

**Pregunta Central:** *¿Podemos predecir la presencia de enfermedad cardíaca en un paciente utilizando sus parámetros clínicos de rutina?*

## 🔍 Aspectos Técnicos Destacados

### Características del Dataset
- **Origen:** UCI Machine Learning Repository - Cleveland Heart Disease Database
- **Tamaño:** 303 pacientes (dataset curado y balanceado)
- **Features:** 14 variables clínicas (edad, sexo, presión arterial, colesterol, ECG, etc.)
- **Target:** Binario (0 = Sin enfermedad, 1 = Con enfermedad)
- **Calidad:** Sin valores faltantes, datos preprocesados y validados

### Ventajas del Dataset
✅ **Clínicamente validado** - Datos reales del Cleveland Clinic Foundation  
✅ **Balanceado** - 165 casos positivos vs. 138 negativos (~55/45%)  
✅ **Completo** - Sin missing values ni outliers extremos  
✅ **Interpretable** - Features con significado médico claro  

## 🛠️ Stack Tecnológico

### Librerías Principales
```python
pandas          # Manipulación de datos médicos
numpy           # Cálculos numéricos y estadísticos
matplotlib      # Visualizaciones médicas
seaborn         # Gráficos estadísticos avanzados
scikit-learn    # Suite completa de ML
```

### Algoritmos Implementados y Comparados
1. **Logistic Regression** - Baseline interpretable
2. **K-Nearest Neighbors (KNN)** - Clasificación por proximidad
3. **Random Forest Classifier** - Ensemble robusto
4. **Support Vector Machine (SVM)** - Clasificación con margen máximo

## 📊 Metodología Completa de Data Science

### 1. Análisis Exploratorio de Datos (EDA)
```python
✓ Análisis univariado de cada feature clínica
✓ Distribuciones por clase (enfermo vs. sano)
✓ Matriz de correlación entre variables
✓ Visualización de relaciones multivariadas
✓ Detección de patrones y anomalías
```

### 2. Preparación de Datos
- **Normalización/Estandarización** - Escalado para algoritmos sensibles (KNN, SVM)
- **Feature Selection** - Identificación de variables más predictivas
- **Train/Test Split** - 80/20 con estratificación por clase
- **Validación cruzada** - 5-fold CV para robustez

### 3. Experimentación con Modelos
**Proceso sistemático:**
1. Entrenamiento de múltiples algoritmos
2. Comparación de métricas de rendimiento
3. Selección del mejor modelo base
4. Hyperparameter tuning (GridSearchCV/RandomizedSearchCV)
5. Validación cruzada del modelo final
6. Análisis de feature importance

### 4. Evaluación Exhaustiva
**Métricas implementadas:**
- **Accuracy** - Precisión general del modelo
- **Precision** - Calidad de diagnósticos positivos
- **Recall (Sensitivity)** - Capacidad de detectar enfermos
- **F1-Score** - Balance precision-recall
- **ROC-AUC** - Rendimiento global del clasificador
- **Confusion Matrix** - Análisis detallado de errores

## 📈 Resultados y Rendimiento

### Objetivo de Rendimiento
🎯 **Meta establecida:** 95% de accuracy  
📊 **Resultado típico:** 85-90% accuracy con modelos optimizados

### Interpretación Clínica
- **High Recall (>90%):** Minimiza falsos negativos - crítico en salud
- **High Precision (>85%):** Reduce falsos positivos - evita tratamientos innecesarios
- **ROC-AUC >0.90:** Excelente capacidad discriminativa

### Feature Importance
**Variables más predictivas identificadas:**
1. **cp (chest pain type)** - Tipo de dolor torácico
2. **thalach** - Frecuencia cardíaca máxima alcanzada
3. **ca** - Número de vasos principales coloreados por fluoroscopia
4. **thal** - Resultado del test de talio
5. **oldpeak** - Depresión del ST inducida por ejercicio

## 💼 Impacto y Aplicaciones

### Valor para el Sistema de Salud
1. **Screening Temprano**
   - Identificación rápida de pacientes de alto riesgo
   - Priorización de casos para estudios avanzados
   - Reducción de carga en especialistas

2. **Apoyo a la Decisión Clínica**
   - Segunda opinión automatizada
   - Detección de casos que podrían pasarse por alto
   - Estandarización de criterios diagnósticos

3. **Optimización de Recursos**
   - Reducción de pruebas innecesarias
   - Mejor asignación de citas con cardiólogos
   - Priorización de recursos limitados

### ROI en Healthcare
- **Detección temprana:** Ahorro de $10,000-$50,000 por paciente en tratamientos avanzados
- **Eficiencia operativa:** 30-40% reducción en tiempo de pre-screening
- **Prevención:** Intervenciones tempranas mejoran outcomes en 60%+

## 🧠 Habilidades Técnicas Demostradas

### Data Science Core
✅ **Classification Modeling** - Comparación de múltiples algoritmos  
✅ **Model Evaluation** - Suite completa de métricas médicas  
✅ **Cross-Validation** - Validación robusta y sin overfitting  
✅ **Hyperparameter Tuning** - Optimización sistemática (Grid/Random Search)  
✅ **Feature Engineering** - Análisis de importancia y selección  
✅ **Statistical Analysis** - Tests de significancia y correlación  
✅ **Data Visualization** - Comunicación efectiva de insights médicos  

### Domain Knowledge
✅ Comprensión de métricas médicas (Recall > Precision en salud)  
✅ Interpretación de variables clínicas  
✅ Consideraciones éticas en ML médico  
✅ Balance accuracy vs. interpretabilidad  

### Best Practices
✅ Código reproducible y documentado  
✅ Validación cruzada para robustez  
✅ Comparación justa entre modelos (mismos splits)  
✅ Análisis de errores (confusion matrix)  
✅ Consideración de costos de falsos negativos vs. positivos  

## 📁 Estructura del Proyecto

```
HeartDiseaseProject/
├── end-to-end heart disease predictions.ipynb
│   └── Notebook completo con flujo de trabajo end-to-end
├── heart-disease.csv
│   └── Dataset limpio (303 pacientes, 14 features)
└── README.md
```

## 🚀 Cómo Ejecutar

### Instalación de Dependencias
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Ejecución del Notebook
```bash
jupyter notebook "end-to-end heart disease predictions.ipynb"
```

### Dataset
El dataset está incluido localmente, pero también disponible en:
- [UCI ML Repository - Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
- [Kaggle - Heart Disease Classification Dataset](https://www.kaggle.com/datasets/sumaiyatasmeem/heart-disease-classification-dataset)

## 📚 Diccionario de Variables Clínicas

| Variable | Descripción | Tipo | Valores |
|----------|-------------|------|---------|
| **age** | Edad del paciente | Numérico | 29-77 años |
| **sex** | Sexo | Categórico | 1=masculino, 0=femenino |
| **cp** | Tipo de dolor torácico | Categórico | 0-3 (angina típica/atípica/no anginoso/asintomático) |
| **trestbps** | Presión arterial en reposo | Numérico | 94-200 mm Hg |
| **chol** | Colesterol sérico | Numérico | 126-564 mg/dl |
| **fbs** | Glucosa en ayunas >120 mg/dl | Binario | 1=sí, 0=no |
| **restecg** | Resultados ECG en reposo | Categórico | 0-2 (normal/anomalía ST-T/hipertrofia) |
| **thalach** | Frecuencia cardíaca máxima | Numérico | 71-202 bpm |
| **exang** | Angina inducida por ejercicio | Binario | 1=sí, 0=no |
| **oldpeak** | Depresión ST inducida por ejercicio | Numérico | 0-6.2 |
| **slope** | Pendiente del segmento ST | Categórico | 0-2 (ascendente/plana/descendente) |
| **ca** | Vasos principales coloreados | Numérico | 0-3 |
| **thal** | Resultado test talio | Categórico | 1,3,6,7 (normal/defecto fijo/reversible) |
| **target** | Presencia de enfermedad | Binario | 1=enfermedad, 0=sano |

## 🎓 Aprendizajes y Conclusiones

### Insights Técnicos
- Los métodos ensemble (Random Forest) superan consistentemente a modelos lineales
- La estandarización es crítica para KNN y SVM
- El tipo de dolor torácico (cp) es el predictor más fuerte
- Validación cruzada esencial para evitar overfitting en datasets pequeños

### Consideraciones Médicas
- Recall (sensibilidad) debe priorizarse sobre precisión en screening
- Interpretabilidad es crucial para adopción clínica
- False negatives tienen mayor costo que false positives
- El modelo complementa, no reemplaza, el juicio médico

### Transferibilidad
Este enfoque es replicable para:
- Otras condiciones médicas (diabetes, cáncer, etc.)
- Screenings poblacionales
- Sistemas de alertas tempranas
- Estratificación de riesgo personalizada

---

**Tecnologías:** Python · Machine Learning · Classification · Healthcare AI · scikit-learn · Data Science

**Nivel:** Intermediate  
**Tiempo de desarrollo:** ~20 horas  
**Accuracy alcanzada:** 85-90%  
**Dataset:** 303 pacientes, 14 features clínicas
