from sklearn.metrics import confusion_matrix, classification_report
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools


def votacao_majoritaria_ponderada(svm_preds, cnn_preds):
    # Defina os pesos para cada modelo
    peso_svm = 0.4
    peso_cnn = 0.6

    # Calcule as previsões combinadas por votação majoritária ponderada
    preds_combinadas = (peso_svm * svm_preds + peso_cnn *
                        cnn_preds) / (peso_svm + peso_cnn)

    # Obtenha as classes preditas usando o argmax
    classes_preditas = np.argmax(preds_combinadas, axis=1)

    return classes_preditas


def media_simples(svm_preds, cnn_preds):
    # Calcule as previsões combinadas por média simples
    preds_combinadas = (svm_preds + cnn_preds) / 2.0

    # Obtenha as classes preditas usando o argmax
    classes_preditas = np.argmax(preds_combinadas, axis=1)

    return classes_preditas


def media_ponderada(svm_preds, cnn_preds):
    # Defina os pesos para cada modelo
    peso_svm = 0.4
    peso_cnn = 0.6

    # Calcule as previsões combinadas por média ponderada
    preds_combinadas = (peso_svm * svm_preds + peso_cnn *
                        cnn_preds) / (peso_svm + peso_cnn)

    # Obtenha as classes preditas usando o argmax
    classes_preditas = np.argmax(preds_combinadas, axis=1)

    return classes_preditas


def maiores_valores(svm_preds, cnn_preds):
    # Realize a fusão selecionando os maiores valores entre as previsões de cada modelo
    fusion_preds = np.maximum(svm_preds, cnn_preds)

    # Obtenha as classes preditas usando o argmax
    classes_preditas = np.argmax(fusion_preds, axis=1)

    return classes_preditas


input_shape = (256, 256, 3)
classes = ["beach", "bus", "cafe_restaurant", "car", "city_center", "forest_path", "grocery_store",
           "home", "library", "metro_station", "office", "park", "residential_area", "train", "tram"]


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, fold=0):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.0f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Classe verdadeira')
    plt.xlabel('Classe prevista')

    plt.savefig("fusion/confusion_matrix_fold_"+str(fold)+".png")

    plt.close()


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
X = []
y = []

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


arq = open("./features_LBP_janela.txt", "r")
for linha in arq:
    aux = linha.split("|")
    lbp_local = []
    for i in range(len(aux)-1):
        classe_local = []
        aux2 = aux[i].split(";")
        for j in aux2:
            classe_local.append(float(j))
        lbp_local.append(classe_local)
    X.append(lbp_local)
    y.append(int(aux[len(aux)-1].replace("\n", "")))
arq.close()

histories = []
scores_array = []
all_true_labels = []
all_predictions = []

acc_votacao_majoritaria = []
acc_media_simples = []
acc_media_ponderada = []
acc_maiores_valores = []

y_preds = []
vetor_resultados_votacao_majoritaria = []
vetor_resultados_media_simples = []
vetor_resultados_media_ponderada = []
vetor_resultados_maiores_valores = []


k_fold = 5
epochs = 2
batch_size = 32
overall_confusion_matrix = np.zeros((len(classes), len(classes)))

best_model, best_acc = None, 0.0

kfold = KFold(n_splits=k_fold, shuffle=True)

fold_no = 1
for train, test in kfold.split(images, labels):
    model = cnn_model()

    print('------------------------------------------------------------------------')
    print(f'Treinamento para o fold {fold_no} ...')

    x_train_cnn = []
    y_train_cnn = []

    x_train_svm = []
    y_train_svm = []

    x_test_cnn = []
    y_test_cnn = []

    x_test_svm = []
    y_test_svm = []

    for i in train:
        for j in range(6):
            x_train_cnn.append(images[i][j])
            y_train_cnn.append(labels[i])

            x_train_svm.append(X[i][j])
            y_train_svm.append(y[i])

    for i in test:
        for j in range(6):
            x_test_cnn.append(images[i][j])
            y_test_cnn.append(labels[i])

            x_test_svm.append(X[i][j])
            y_test_svm.append(y[i])

    x_train_cnn = np.array(x_train_cnn)
    x_test_cnn = np.array(x_test_cnn)
    y_train_cnn = np.array(y_train_cnn)
    y_test_cnn = np.array(y_test_cnn)

    y_pred = y_test_cnn[:]
    y_preds = np.concatenate((y_preds, y_pred))

    y_train_cnn = to_categorical(y_train_cnn, num_classes=len(classes))
    y_test_cnn = to_categorical(y_test_cnn, num_classes=len(classes))

    # SVM
    svm_model = SVC(C=100, kernel='poly', gamma='scale', probability=True)
    svm_model.fit(x_train_svm, y_train_svm)
    svm_predictions = svm_model.predict_proba(x_test_svm)

    # CNN
    scores = model.evaluate(x_test_cnn, y_test_cnn, verbose=0)
    histories.append(model.history.history)
    cnn_predictions = model.predict(x_test_cnn)

    resultado_votacao_majoritaria = votacao_majoritaria_ponderada(
        svm_predictions, cnn_predictions)
    resultado_media_simples = media_simples(
        svm_predictions, cnn_predictions)
    resultado_media_ponderada = media_ponderada(
        svm_predictions, cnn_predictions)
    resultado_maiores_valores = maiores_valores(
        svm_predictions, cnn_predictions)

    for res in resultado_votacao_majoritaria:
        vetor_resultados_votacao_majoritaria.append(res)

    for res in resultado_media_simples:
        vetor_resultados_media_simples.append(res)

    for res in resultado_media_ponderada:
        vetor_resultados_media_ponderada.append(res)

    for res in resultado_maiores_valores:
        vetor_resultados_maiores_valores.append(res)

    acc_aux = accuracy_score(y_pred, resultado_votacao_majoritaria)
    acc_votacao_majoritaria.append(acc_aux)
    print("acc_votacao_majoritaria:", acc_aux)

    acc_aux = accuracy_score(y_pred, resultado_media_simples)
    acc_media_simples.append(acc_aux)
    print("acc_media_simples:", acc_aux)

    acc_aux = accuracy_score(y_pred, resultado_media_ponderada)
    acc_media_ponderada.append(acc_aux)
    print("acc_media_ponderada:", acc_aux)

    acc_aux = accuracy_score(y_pred, resultado_maiores_valores)
    acc_maiores_valores.append(acc_aux)
    print("acc_maiores_valores:", acc_aux)

    fold_no += 1

print("Fusão")
lista_acuracia = [np.mean(acc_votacao_majoritaria), np.mean(acc_media_simples),
                  np.mean(acc_media_ponderada), np.mean(acc_maiores_valores)]
nomes_metodos = ['Votação Majoritária', 'Média Simples',
                 'Média Ponderada', 'Maiores Valores']

for i in range(len(nomes_metodos)):
    print(f"Acurácia média para {nomes_metodos[i]}: {lista_acuracia[i]}")

melhor_metodo = np.argmax(lista_acuracia)
melhor_acuracia = lista_acuracia[melhor_metodo]
nome_melhor_metodo = nomes_metodos[melhor_metodo]

print("Melhor método", nome_melhor_metodo)
resultado = []
if melhor_metodo == 0:
    resultado = vetor_resultados_votacao_majoritaria[:]
elif melhor_metodo == 1:
    resultado = vetor_resultados_media_simples[:]
elif melhor_metodo == 2:
    resultado = vetor_resultados_media_ponderada[:]
elif melhor_metodo == 3:
    resultado = vetor_resultados_maiores_valores[:]

matriz_confusao = confusion_matrix(y_preds, resultado)
relatorio_classificacao = classification_report(y_preds, resultado)

plot_confusion_matrix(matriz_confusao, classes=[
    i for i in range(1, 16)], title='Matriz geral')

print("Tabela F1-score - {}".format(nome_melhor_metodo))
print(relatorio_classificacao)
