from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import KFold
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools


input_shape = (256, 256, 3)
classes = ["beach", "bus", "cafe_restaurant", "car", "city_center", "forest_path", "grocery_store",
           "home", "library", "metro_station", "office", "park", "residential_area", "train", "tram"]


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, fold=1):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Classe verdadeira')
    plt.xlabel('Classe prevista')

    plt.savefig("saida/confusion_matrix_fold_"+str(fold)+".png")


def cnn_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                               input_shape=input_shape, padding='same'),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((3, 3)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((3, 3)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((3, 3)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(classes), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


labels = []
images = []

path = './spec_janela2/'

max_imagens = 1170

j = 0
k = 0
for i in range(1170):
    aux = []
    if j == 78:
        j = 0
        k += 1
    path_total = path + 'class_' + classes[k] + '/' + str(i) + '/'
    for l in range(6):
        path_img = path_total + str(l) + '.png'
        img = cv2.imread(path_img)
        img = cv2.resize(img, (256, 256))
        aux.append(img)
    images.append(aux)
    labels.append(k)
    j += 1

acc_per_fold = []
loss_per_fold = []
histories = []
scores_array = []
confusion_matrices = []

k_fold = 5
epochs = 2  # 20
batch_size = 32

best_model, best_acc = None, 0.0

kfold = KFold(n_splits=k_fold, shuffle=True)

fold_no = 1
for train, test in kfold.split(images, labels):
    model = cnn_model()

    print('------------------------------------------------------------------------')
    print(f'Treinamento para o fold {fold_no} ...')

    x_train = []
    y_train = []

    x_test = []
    y_test = []

    for i in train:
        for j in range(6):
            x_train.append(images[i][j])
            y_train.append(labels[i])

    for i in test:
        for j in range(6):
            x_test.append(images[i][j])
            y_test.append(labels[i])

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    y_pred = y_test[:]

    y_train = to_categorical(y_train, num_classes=len(classes))
    y_test = to_categorical(y_test, num_classes=len(classes))

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs)

    histories.append(history.history)

    scores = model.evaluate(x_test, y_test, verbose=0)

    if scores[1] > best_acc:
        best_acc = scores[1]
        best_model = model
    scores_array.append(scores)
    print(
        f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    current_fold_predictions = model.predict(x_test)
    current_fold_predictions = np.argmax(current_fold_predictions, axis=1)

    confusion_matrix_fold = confusion_matrix(
        y_pred, current_fold_predictions)

    # plt.figure(figsize=(8, 6))
    # plt.imshow(confusion_matrix_fold,
    #            interpolation='nearest', cmap=plt.cm.Blues)
    # plt.title(f'Matriz de Confusão - Fold {fold_no}')
    # plt.colorbar()
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    # plt.yticks(tick_marks, classes)
    # plt.xlabel('Rótulo Predito')
    # plt.ylabel('Rótulo Verdadeiro')

    # plt.savefig("saida/confusion_matrix_fold_"+str(fold_no)+".png")
    # plt.close()

    plot_confusion_matrix(confusion_matrix_fold, classes=[
                          i for i in range(1, 16)], title='Matriz de confusão fold '+str(fold_no), fold=fold_no)

    fold_no += 1


# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
    print('------------------------------------------------------------------------')
    print(
        f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')


loss = []
accuracy = []

for i in range(k_fold):
    aux_loss = histories[i]['loss']
    aux_accuracy = histories[i]['accuracy']

    loss.append(aux_loss)
    accuracy.append(aux_accuracy)


plt.figure(figsize=(15, 5))

plt_loss = plt.subplot(121)
for fold in range(len(loss)):
    plt.plot(loss[fold], label=f'fold {fold+1}')
plt.title("Perda")
plt.ylabel("Perda")
plt.xlabel("Época")
plt.legend()

plt_accuracy = plt.subplot(122)
for fold in range(len(accuracy)):
    plt.plot(accuracy[fold], label=f'fold {fold+1}')
plt.title("Acurácia")
plt.ylabel("Acurácia")
plt.xlabel("Época")
plt.legend()

# Salvar os gráficos em formato PNG
plt.savefig("saida/graficos.png")

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.boxplot(loss)
plt.title('Validação Perda')
plt.xlabel('fold')
plt.ylabel('Perda')

plt.subplot(1, 2, 2)
plt.boxplot(accuracy)
plt.title('Validação Acurácia')
plt.xlabel('fold')
plt.ylabel('Acurácia')

plt.tight_layout()

plt.savefig("saida/boxplot.png")

model_json = model.to_json()
with open("saida/model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("saida/model.h5")
print("Saved model to disk")
