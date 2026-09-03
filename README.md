# Regresión Logística desde Cero 🧠

Implementación de un modelo de **regresión logística binaria** construido desde cero con NumPy (sin usar `scikit-learn` para el entrenamiento), aplicado al dataset **Breast Cancer Wisconsin** para clasificar tumores como malignos o benignos.

El notebook incluye, además, una comparación directa contra `LogisticRegression` de `scikit-learn` para validar que la implementación converge a los mismos resultados.

## 📊 Dataset

Se utiliza el dataset [Breast Cancer Wisconsin](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset), incluido en `sklearn.datasets`.

- **569 observaciones**, 30 características originales.
- Para poder visualizar la frontera de decisión en 2D, el modelo se entrena únicamente con dos variables:
  - `mean radius`
  - `mean texture`
- Variable objetivo: `0` = Maligno, `1` = Benigno.

## ⚙️ ¿Qué hace el notebook?

1. **Carga y exploración** del dataset.
2. **Selección de variables** (`mean radius`, `mean texture`) y **train/test split** (80/20).
3. **Visualización** de la dispersión de clases.
4. **Estandarización** de las variables con `StandardScaler` (clave para que el descenso de gradiente converja bien).
5. **Implementación desde cero** de:
   - `logistic_regresion_fit`: entrena los coeficientes (`beta`) mediante **descenso de gradiente**, usando la función sigmoide y **log-loss** como función de costo.
   - `logistic_regresion_predict`: genera predicciones binarias a partir de las probabilidades (umbral = 0.5).
6. **Evaluación** con matriz de confusión.
7. **Comparación** contra `LogisticRegression(penalty=None)` de scikit-learn.

## 📈 Resultados

El modelo propio converge prácticamente a los mismos coeficientes que scikit-learn:

| Parámetro          | Modelo propio | scikit-learn |
|---------------------|---------------:|-------------:|
| Intercepto (β₀)      |         0.8405 |       0.8501 |
| mean radius (β₁)     |        -3.4615 |      -3.6161 |
| mean texture (β₂)    |        -0.9165 |      -0.9391 |

## 🧩 Estructura del notebook

- Importación de librerías
- Carga del dataset
- Exploración de los datos
- Selección de variables
- División en train/test
- Diagrama de dispersión
- Estandarización de variables
- Implementación de regresión logística (fit / predict)
- Ajuste del modelo y predicción
- Matriz de confusión
- Comparación contra scikit-learn

## 🛠️ Tecnologías utilizadas

- Python
- NumPy
- scikit-learn 
- Matplotlib

## 📌 Notas

- El descenso de gradiente funciona mejor con **variables estandarizadas** para converger en un número razonable de épocas; sin estandarizar, el modelo necesita muchas más iteraciones o diverge con learning rates altos.
- Este proyecto tiene fines **educativos**: el objetivo es entender la mecánica interna de la regresión logística (función sigmoide, log-loss, gradiente) en lugar de usarla como una caja negra.

