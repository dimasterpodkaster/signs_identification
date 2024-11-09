import os
import PIL
import matplotlib
import random
import numpy as np
import keras
import tensorflow
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

data = []
labels = []

# Рассмотрим заданный архив знаков
classes = 42
current_directory = os.getcwd()
folder_path = os.path.join(current_directory, 'archive/Train')
print(folder_path)

for num in range(0, classes):
    path = os.path.join(folder_path, str(num))
    imagePaths = os.listdir(path)
    counter = 0
    for img in imagePaths:
        image = Image.open(path + '/' + img)
        if counter == 0:
            pass
            # Можно показать примеры изображений класса
            # plt.imshow(image)
            # plt.axis('off')
            # plt.show()
        image = image.resize((30, 30))
        image = keras.utils.img_to_array(image)
        data.append(image)
        labels.append(num)
        counter += 1

data = np.array(data)
labels = np.array(labels)
print(data.shape, labels.shape)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# Найдем количество изображений в каждом классе
def cnt_img_in_classes(labels_all):
    count = {}
    for i in labels_all:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1
    return count


samples_distribution = cnt_img_in_classes(y_train)
samples_distribution = dict(sorted(samples_distribution.items()))
print(samples_distribution)


def diagram(count_classes):
    plt.bar(range(len(count_classes)), list(count_classes.values()), align='center')
    plt.xticks(range(len(count_classes)), list(count_classes.keys()), rotation=90, fontsize=7)
    plt.show()


diagram(samples_distribution)
# Диаграмма распределения показывает, что обучающий набор данных не сбалансирован


# Будем использовать oversampling и data augmentation
def augment_images(images, batch_size=32):
    """
    Применяет аугментацию ко всем изображениям в массиве.

    :param images: Массив изображений с формой (N, 30, 30, 3), где N - количество изображений.
    :param batch_size: Размер батча для аугментации (по умолчанию 32).
    :return: Массив аугментированных изображений с той же формой (N, 30, 30, 3).
    """
    # Создаем объект ImageDataGenerator с нужными аугментациями
    datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Преобразуем входные данные в нужный формат (батч из изображений)
    images = np.array(images)  # Если изображения передаются не как numpy массив
    images = images.astype('float32')  # Убедитесь, что тип данных корректен для аугментации

    # Создаем генератор для аугментации изображений
    datagen.fit(images)

    # Инициализируем пустой список для хранения аугментированных изображений
    augmented_images = []

    # Генерация аугментированных изображений
    # batch_size=1, чтобы генерировать по одному изображению за раз
    for augmented_batch in datagen.flow(images, batch_size=batch_size, shuffle=False):
        # Мы знаем, что в одном батче будет batch_size изображений
        augmented_images.append(augmented_batch)

        # Прерываем, если мы обработали все изображения
        if len(augmented_images) * batch_size >= len(images):
            break

    # Конкатенируем все аугментированные изображения в один массив
    augmented_images = np.concatenate(augmented_images, axis=0)

    # Визуализация одного случайного аугментированного изображения
    # example_image = augmented_images[0]  # Возьмем первое изображение как пример
    # plt.imshow(example_image.astype('uint8'))
    # plt.axis('off')  # Убираем оси для удобства
    # plt.show()

    return augmented_images


def oversampling_or_augmentation(images, labels_all, method):
    min_imgs = 500
    classes_all = cnt_img_in_classes(
        labels_all)  # предположим, что эта функция возвращает количество изображений в каждом классе

    for i in range(len(classes_all)):
        print('I is', i)
        if classes_all[i] < min_imgs:
            add_num = min_imgs - classes_all[i]
            imgs_for_augm = []
            lbls_for_augm = []

            for j in range(add_num):
                im_index = random.choice(np.where(labels_all == i)[0])
                imgs_for_augm.append(images[im_index])
                lbls_for_augm.append(labels_all[im_index])

            if method == "augmentation":
                augmented_class = augment_images(imgs_for_augm)
                concat_imgs_np = np.array(augmented_class)
            else:
                concat_imgs_np = np.array(imgs_for_augm)

            concat_lbls_np = np.array(lbls_for_augm)
            print("LENGTH of concat_imgs_np:", len(concat_imgs_np))
            print("LENGTH of concat_lbls_np:", len(concat_lbls_np))

            # Объединяем исходные и новые изображения и метки
            images = np.concatenate((images, concat_imgs_np), axis=0)
            labels_all = np.concatenate((labels_all, concat_lbls_np), axis=0)

    return images, labels_all


X_train, y_train = oversampling_or_augmentation(X_train, y_train, 'augmentation')
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

augmented_samples_distribution = cnt_img_in_classes(y_train)
augmented_samples_distribution = dict(sorted(augmented_samples_distribution.items()))
diagram(augmented_samples_distribution)
# Набор стал более сбалансирован

# Преобразование меток, присутствующих в y_train и y_test в one-hot encoding.
y_train = keras.utils.to_categorical(y_train, 43)
y_test = keras.utils.to_categorical(y_test, 43)


def create_cnn_model(input_shape=(30, 30, 3), num_classes=43):
    model = keras.models.Sequential()

    # Первый блок свертки и подвыборки
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))  # Регуляризация Dropout

    # Второй блок свертки и подвыборки
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Третий блок свертки и подвыборки
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    # Полносвязный слой
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    # Выходной слой
    model.add(Dense(num_classes, activation='softmax'))

    return model


# Инициализация модели
model = create_cnn_model(input_shape=(30, 30, 3), num_classes=43)

# Компиляция модели
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Добавим раннюю остановку для контроля обучения
early_stopping = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Инициализируем генератор только для нормализации
datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

# Создаём генераторы для обучающих и валидационных данных
train_generator = datagen.flow(X_train, y_train, batch_size=32, shuffle=True)
validation_generator = datagen.flow(X_test, y_test, batch_size=32, shuffle=False)

# Пример использования генераторов при обучении модели
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=30,
    callbacks=[early_stopping]
)

# С помощью matplotlib строим график для точности и потерь
plt.style.use("ggplot")
plt.figure()
N = len(history.history["loss"])
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()

# Сохранение модели целиком
model.save("signs_identification_model.h5")

# Сохранение x_test и y_test в отдельных файлах .npy
np.save("x_test.npy", X_test)
np.save("y_test.npy", y_test)
