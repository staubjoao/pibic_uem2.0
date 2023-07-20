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
import pickle


def votacao(svm_preds, cnn_preds):
    fused_predictions = []
    # Obtém o número de classes do vetor de previsões do CNN
    num_classes = len(cnn_predictions[0])

    for svm_pred, cnn_pred in zip(svm_predictions, cnn_predictions):
        # Contando os votos para cada classe
        votes = np.zeros(num_classes, dtype=int)

        # Incrementando o voto do SVM
        svm_pred = int(svm_pred)
        votes[svm_pred] += 1

        # Incrementando os votos do CNN
        max_class_index = np.argmax(cnn_pred)
        votes[max_class_index] += 1

        # Obtendo a classe com mais votos
        fused_pred = np.argmax(votes)
        fused_predictions.append(fused_pred)

    return fused_predictions


def soma_simples(svm_preds, cnn_preds):
    preds_compinadas = svm_preds + cnn_preds

    return np.argmax(preds_compinadas, axis=1)


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

    plt.savefig("fusion2/confusion_matrix_fold_"+str(fold)+".png")

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

# Leitura dos dados para a CNN
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

# Leitura dos dados para SVM
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
acc_soma_simples = []
acc_media_simples = []
acc_media_ponderada = []
acc_maiores_valores = []
vetor_resultados_soma_simples = []
vetor_resultados_media_simples = []
vetor_resultados_media_ponderada = []
vetor_resultados_maiores_valores = []
y_preds = []

svm_predictions_fold = np.array([])
cnn_predictions_fold = np.array([])
y_pred_fold = np.array([])


# Definição dos valores para treinamento
k_fold = 5
epochs = 2
batch_size = 64

# Para a CNN
acc_per_fold = []
loss_per_fold = []
best_model, best_acc = None, 0.0
best_model_svm, best_acc_svm = None, 0.0

# Split dos folds
kfold = KFold(n_splits=k_fold, shuffle=True)

# For para treinamento entre os folds
fold_no = 1
for train, test in kfold.split(images, labels):
    # Criação do modelo
    model = cnn_model()

    print('------------------------------------------------------------------------')
    print(f'Treinamento para o fold {fold_no} ...')

    # Vetores para teste e treinamento
    x_train_cnn = []
    y_train_cnn = []

    x_train_svm = []
    y_train_svm = []

    x_test_cnn = []
    y_test_cnn = []

    x_test_svm = []
    y_test_svm = []

    # Fazendo o split dos dados para treinamento
    for i in train:
        for j in range(6):
            x_train_cnn.append(images[i][j])
            y_train_cnn.append(labels[i])

            x_train_svm.append(X[i][j])
            y_train_svm.append(y[i])

    # Fazendo o split dos dados para teste
    for i in test:
        for j in range(6):
            x_test_cnn.append(images[i][j])
            y_test_cnn.append(labels[i])

            x_test_svm.append(X[i][j])
            y_test_svm.append(y[i])

    # Tranformando em vetores numpy
    x_train_cnn = np.array(x_train_cnn)
    x_test_cnn = np.array(x_test_cnn)
    y_train_cnn = np.array(y_train_cnn)
    y_test_cnn = np.array(y_test_cnn)

    y_pred = y_test_cnn[:]
    for res in y_pred:
        y_preds.append(res)

    y_pred_fold.append(y_pred)

    y_train_cnn = to_categorical(y_train_cnn, num_classes=len(classes))
    y_test_cnn = to_categorical(y_test_cnn, num_classes=len(classes))

    # SVM
    svm_model = SVC(C=100, kernel='poly', gamma='scale', probability=True)
    svm_model.fit(x_train_svm, y_train_svm)
    svm_predictions = svm_model.predict_proba(x_test_svm)

    svm_predictions_fold.append(svm_predictions)

    svm_predictions_acc = svm_model.predict(x_test_svm)
    acc_svm = accuracy_score(y_test_svm, svm_predictions_acc)

    if acc_svm > best_acc_svm:
        best_acc_svm = acc_svm
        best_model_svm = svm_model

    # CNN
    history = model.fit(x_train_cnn, y_train_cnn,
                        batch_size=batch_size,
                        epochs=epochs)
    histories.append(history.history)
    scores = model.evaluate(x_test_cnn, y_test_cnn, verbose=0)
    cnn_predictions = model.predict(x_test_cnn)

    cnn_predictions_fold.append(cnn_predictions)

    # Acurácia para cada fusão
    resultado_soma_simples = soma_simples(
        svm_predictions, cnn_predictions)
    resultado_media_simples = media_simples(
        svm_predictions, cnn_predictions)
    resultado_media_ponderada = media_ponderada(
        svm_predictions, cnn_predictions)
    resultado_maiores_valores = maiores_valores(
        svm_predictions, cnn_predictions)

    # Adiciona o resultado no vetor para fazer a validação fora do for
    for res in resultado_soma_simples:
        vetor_resultados_soma_simples.append(res)

    for res in resultado_media_simples:
        vetor_resultados_media_simples.append(res)

    for res in resultado_media_ponderada:
        vetor_resultados_media_ponderada.append(res)

    for res in resultado_maiores_valores:
        vetor_resultados_maiores_valores.append(res)

    # Imprime os resultados
    acc_aux = accuracy_score(y_pred, vetor_resultados_soma_simples)
    acc_soma_simples.append(acc_aux)
    print("acc_soma_simples:", acc_aux)

    acc_aux = accuracy_score(y_pred, resultado_media_simples)
    acc_media_simples.append(acc_aux)
    print("acc_media_simples:", acc_aux)

    acc_aux = accuracy_score(y_pred, resultado_media_ponderada)
    acc_media_ponderada.append(acc_aux)
    print("acc_media_ponderada:", acc_aux)

    acc_aux = accuracy_score(y_pred, resultado_maiores_valores)
    acc_maiores_valores.append(acc_aux)
    print("acc_maiores_valores:", acc_aux)

    # Recolhe as melhores accuracias no modelo CNN para salvar depois
    if scores[1] > best_acc:
        best_acc = scores[1]
        best_model = model
    scores_array.append(scores)
    print(
        f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    fold_no += 1

# Salvar predições em texto
print("y_pred_fold:")
for i in y_pred_fold:
    print(i)

print("svm_predictions_fold:")
for i in svm_predictions_fold:
    print(i)

print("cnn_predictions_fold:")
for i in cnn_predictions_fold:
    print(i)

np.savetxt('saida/predicoes_real,txt', y_pred_fold, fmt="%f")
np.savetxt('saida2/predicoes_svm.txt', svm_predictions_fold, fmt="%f")
np.savetxt('saida2/predicoes_cnn.txt', cnn_predictions_fold, fmt="%f")


# Salvar modelo SVM
with open('saida2/modelo_svm.pkl', 'wb') as f:
    pickle.dump(svm_model, f)

# Imprime os resultados
print("Fusão")
lista_acuracia = [np.mean(acc_soma_simples), np.mean(acc_media_simples),
                  np.mean(acc_media_ponderada), np.mean(acc_maiores_valores)]
nomes_metodos = ['Soma Simples', 'Média Simples',
                 'Média Ponderada', 'Maiores Valores']

# Acurácia média para cada fusão
for i in range(len(nomes_metodos)):
    print(f"Acurácia média para {nomes_metodos[i]}: {lista_acuracia[i]}")

# Armazena o melhor método de fusão
melhor_metodo = np.argmax(lista_acuracia)
melhor_acuracia = lista_acuracia[melhor_metodo]
nome_melhor_metodo = nomes_metodos[melhor_metodo]

# Imprime e plota a matriz e o f1-score
print("Melhor método", nome_melhor_metodo)
resultado = []
if melhor_metodo == 0:
    resultado = acc_soma_simples[:]
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

loss = []
accuracy = []

# Para plotar os graficos para CNN
for i in range(k_fold):
    aux_loss = histories[i]['loss']
    aux_accuracy = histories[i]['accuracy']

    loss.append(aux_loss)
    accuracy.append(aux_accuracy)

# Plot dos graficos da CNN
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
plt.savefig("saida2/graficos.png")

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
plt.savefig("saida2/boxplot.png")

# Salvar modelo
model_json = model.to_json()
with open("saida2/model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("saida/model.h5")
print("Saved model to disk")
