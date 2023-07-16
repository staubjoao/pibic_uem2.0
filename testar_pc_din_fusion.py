from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools


def fuse_predictions_voting(svm_predictions, cnn_predictions):
    fused_predictions = []
    num_classes = len(cnn_predictions[0])

    for svm_pred, cnn_pred in zip(svm_predictions, cnn_predictions):
        votes = np.zeros(num_classes, dtype=int)

        svm_pred = int(svm_pred)
        votes[svm_pred] += 1

        max_class_index = np.argmax(cnn_pred)
        votes[max_class_index] += 1

        fused_pred = np.argmax(votes)
        fused_predictions.append(fused_pred)

    return fused_predictions


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

    plt.savefig("saida/confusion_matrix_fold_"+str(fold)+".png")

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

acc_per_fold_cnn = []
acc_per_fold_svm = []
acc_per_fold_fusion = []
loss_per_fold_cnn = []
loss_per_fold_svm = []
loss_per_fold_fusion = []
histories = []
scores_array = []
all_true_labels = []
all_predictions = []


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
    # print("y_pred: ", y_pred)

    y_train_cnn = to_categorical(y_train_cnn, num_classes=len(classes))
    y_test_cnn = to_categorical(y_test_cnn, num_classes=len(classes))

    # SVM
    svm_model = SVC(C=100, kernel='poly', gamma='scale', probability=True)
    svm_model.fit(x_train_svm, y_train_svm)
    # svm_predictions_prob = svm_model.predict_proba(x_test_svm)
    # svm_predictions = np.argmax(svm_predictions_prob, axis=1)
    svm_predictions = svm_model.predict(x_test_svm)
    accuracy_svm = accuracy_score(y_pred, svm_predictions)
    # loss_svm = log_loss(y_pred, svm_predictions, labels=range(15))

    acc_per_fold_svm.append(accuracy_svm)
    # loss_per_fold_svm.append(loss_svm)

    # CNN
    scores = model.evaluate(x_test_cnn, y_test_cnn, verbose=0)
    histories.append(model.history.history)
    cnn_predictions = model.predict(x_test_cnn)
    if len(cnn_predictions.shape) > 1:
        cnn_predictions = np.argmax(cnn_predictions, axis=1)
    accuracy_cnn = accuracy_score(y_pred, cnn_predictions)
    # loss_cnn = log_loss(y_pred, cnn_predictions, labels=range(15))

    acc_per_fold_cnn.append(accuracy_cnn)
    # loss_per_fold_cnn.append(loss_cnn)

    fused_predictions_voting = fuse_predictions_voting(
        svm_predictions, cnn_predictions)

    all_true_labels.extend(y_pred)
    all_predictions.extend(fused_predictions_voting)

    # Calcula a acurácia e perda da fusão por votação
    accuracy_fusion = accuracy_score(y_pred, fused_predictions_voting)
    loss = log_loss(y_pred, fused_predictions_voting)

    # Armazena a acurácia e perda
    acc_per_fold_fusion.append(accuracy_fusion)
    loss_per_fold_fusion.append(loss)

    # Calcula a matriz de confusão e acumula na matriz overall_confusion_matrix
    fold_confusion_matrix = confusion_matrix(y_pred, fused_predictions_voting)
    overall_confusion_matrix += fold_confusion_matrix

    # Imprime os resultados para cada fold
    print(f'Acurácia do fold {fold_no}: {accuracy_fusion}')
    print(
        f'Acurácia da fusão por votação do fold {fold_no}: {accuracy_fusion}')
    print(f'Perda da fusão por votação do fold {fold_no}: {loss}')

    fold_no += 1

print("Fusão")
for i, accuracy in enumerate(acc_per_fold_fusion):
    print(f"Fold {i+1} - Accuracy: {accuracy}")

avg_accuracy_fusion = np.mean(acc_per_fold_fusion)
print(f"Média accuracy: {avg_accuracy_fusion}")

for i, class_name in enumerate(classes):
    true_count = np.sum(overall_confusion_matrix[i, :])
    print(f'Classe {class_name}: Total real = {true_count}')

plot_confusion_matrix(overall_confusion_matrix, classes=[
    i for i in range(1, 16)], title='Matriz geral')

plt.plot(range(1, k_fold + 1), acc_per_fold_svm, marker='o')
plt.plot(range(1, k_fold + 1), acc_per_fold_cnn, marker='o')
plt.plot(range(1, k_fold + 1), acc_per_fold_fusion, marker='o')
plt.xlabel('Fold')
plt.ylabel('Acurácia')
plt.legend(['SVM', 'CNN', 'Fusão por Votação'])
plt.title('Acurácia por Fold')
plt.savefig("fusion/acuracia_svm_cnn_fusao.png")
plt.close()


# plt.plot(range(1, k_fold + 1), loss_per_fold_svm, marker='o')
# plt.plot(range(1, k_fold + 1), loss_per_fold_cnn, marker='o')
# plt.plot(range(1, k_fold + 1), loss_per_fold_fusion, marker='o')
# plt.xlabel('Fold')
# plt.ylabel('Acurácia')
# plt.legend(['SVM', 'CNN', 'Fusão por Votação'])
# plt.title('Perda por Fold')
# plt.savefig("fusion/perda_svm_cnn_fusao.png")
# plt.close()


fpr, tpr, _ = roc_curve(all_true_labels, all_predictions)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label='Curva ROC (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC para Todos os Folds')
plt.legend(loc='lower right')
plt.savefig("fusion/roc_curve.png")
plt.close()

# # == Provide average scores ==
# print('------------------------------------------------------------------------')
# print('Score per fold')
# for i in range(0, len(acc_per_fold)):
#     print('------------------------------------------------------------------------')
#     print(
#         f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
# print('------------------------------------------------------------------------')
# print('Average scores for all folds:')
# print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
# print(f'> Loss: {np.mean(loss_per_fold)}')
# print('------------------------------------------------------------------------')


# # loss = []
# accuracy = []

# for i in range(k_fold):
#     # aux_loss = histories[i]['loss']
#     aux_accuracy = histories[i]['accuracy']

#     # loss.append(aux_loss)
#     accuracy.append(aux_accuracy)


# plt.figure(figsize=(15, 5))

# # plt_loss = plt.subplot(121)
# # for fold in range(len(loss)):
# #     plt.plot(loss[fold], label=f'fold {fold+1}')
# # plt.title("Perda")
# # plt.ylabel("Perda")
# # plt.xlabel("Época")
# # plt.legend()

# plt_accuracy = plt.subplot(122)
# for fold in range(len(accuracy)):
#     plt.plot(accuracy[fold], label=f'fold {fold+1}')
# plt.title("Acurácia")
# plt.ylabel("Acurácia")
# plt.xlabel("Época")
# plt.legend()

# # Salvar os gráficos em formato PNG
# plt.savefig("saida/graficos.png")

# plt.figure(figsize=(10, 5))

# # plt.subplot(1, 2, 1)
# # plt.boxplot(loss)
# # plt.title('Validação Perda')
# # plt.xlabel('fold')
# # plt.ylabel('Perda')

# plt.subplot(1, 2, 2)
# plt.boxplot(accuracy)
# plt.title('Validação Acurácia')
# plt.xlabel('fold')
# plt.ylabel('Acurácia')

# plt.tight_layout()

# plt.savefig("saida/boxplot.png")

# model_json = model.to_json()
# with open("saida/model.json", "w") as json_file:
#     json_file.write(model_json)

# model.save_weights("saida/model.h5")
# print("Saved model to disk")
