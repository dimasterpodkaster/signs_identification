import tensorflow
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Загрузка всей модели
model = tensorflow.keras.models.load_model("signs_identification_model.h5")

# Загрузка из отдельных файлов .npy
x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")

# Получение предсказаний на тестовых данных
y_pred = model.predict(x_test)

# Классификация, метки были закодированы как one-hot
y_pred_classes = np.argmax(y_pred, axis=1)  # Преобразует вероятности в классы
y_true = np.argmax(y_test, axis=1)  # Истинные классы

# Пример вывода
print(y_pred_classes[:5])  # Печать первых 5 предсказаний
print(y_true[:5])          # Печать первых 5 истинных меток

# С помощью accuracy_score из sklearn metrics проверяем точность предсказаний нашей модели
print(accuracy_score(y_true, y_pred_classes))
# Результат 0.985116756479343

# Выбор случайного индекса из тестовых данных
index = np.random.randint(0, len(x_test))

# Получение изображения и истинного класса
image = x_test[index]
true_label = np.argmax(y_test[index])

# Предсказание класса модели
pred_probabilities = model.predict(np.expand_dims(image, axis=0))  # Добавляем измерение для батча
predicted_label = np.argmax(pred_probabilities)

# Вывод изображения и результатов
plt.imshow(image.astype("uint8"))  # Преобразуем в формат для отображения
plt.title(f"True Label: {true_label}, Predicted Label: {predicted_label}")
plt.axis("off")
plt.show()
