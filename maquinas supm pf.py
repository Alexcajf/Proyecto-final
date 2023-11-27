from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Fetch dataset
zoo = fetch_ucirepo(id=111)

# Data (as pandas dataframes)
X = np.array(zoo.data.features)
y = np.array(zoo.data.targets)

# Metadata
print(zoo.metadata)

# Variable information
print(zoo.variables)

# Asegúrate de que X y y tengan el mismo número de muestras
assert X.shape[0] == y.shape[0]

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un modelo de Máquinas de Soporte Vectorial (SVM)
svm_model = SVC()

# Entrenar el modelo
svm_model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = svm_model.predict(X_test)

# Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:\n', report)
