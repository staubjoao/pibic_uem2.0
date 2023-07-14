from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools


def average_fusion(predictions):
    return np.mean(predictions, axis=0)


def maximum_fusion(predictions):
    return np.max(predictions, axis=0)


def voting_fusion(predictions):
    return np.argmax(np.bincount(predictions))


def calculate_accuracy(true_labels, predictions):
    correct_predictions = np.equal(predictions, true_labels)
    accuracy = np.mean(correct_predictions) * 100.0
    return accuracy


def calculate_loss(true_labels, predictions):
    loss = tf.keras.losses.categorical_crossentropy(true_labels, predictions)
    mean_loss = tf.reduce_mean(loss)
    return mean_loss


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

acc_per_fold = []
acc_per_fold_fusion = []
loss_per_fold = []
histories = []
scores_array = []
confusion_matrices = []

fusion_methods = ['average', 'maximum', 'voting']
k_fold = 5
epochs = 2
batch_size = 32
overall_confusion_matrix = np.zeros((len(classes), len(classes)))

best_model, best_acc = None, 0.0

kfold = KFold(n_splits=k_fold, shuffle=True)

fold_no = 1
accuracy = []  # lista para armazenar as precisões de cada fold
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

    label_encoder = LabelEncoder()
    y_train_cnn = label_encoder.fit_transform(y_train_cnn)
    y_test_cnn = label_encoder.transform(y_test_cnn)

    y_pred = y_test_cnn[:]

    y_train_cnn = to_categorical(y_train_cnn, num_classes=len(classes))
    y_test_cnn = to_categorical(y_test_cnn, num_classes=len(classes))

    # SVM
    svm_model = SVC(C=100, kernel='poly', gamma='scale')
    svm_model.fit(x_train_svm, y_train_svm)
    svm_predictions = svm_model.predict(x_test_svm)
    # print("SVM: ", svm_predictions)

    # CNN
    scores = model.evaluate(x_test_cnn, y_test_cnn, verbose=0)
    histories.append(model.history.history)
    cnn_predictions = model.predict(x_test_cnn)
    # print("CNN: ", svm_predictions)

    # results = {}

    # # fusion
    # for fusion_method in fusion_methods:
    #     if fusion_method == 'average':
    #         fused_predictions = average_fusion(
    #             [cnn_predictions, svm_predictions])
    #     elif fusion_method == 'maximum':
    #         fused_predictions = maximum_fusion(
    #             [cnn_predictions, svm_predictions])
    #     elif fusion_method == 'voting':
    #         fused_predictions = voting_fusion(
    #             [cnn_predictions, svm_predictions])
    #     else:
    #         raise ValueError(
    #             f'Fusion method {fusion_method} is not supported.')
    #     accuracy = calculate_accuracy(y_pred, fused_predictions)
    #     loss = calculate_loss(y_pred, fused_predictions)

    #     results[fusion_method] = {'accuracy': accuracy, 'loss': loss}

    fusion_predictions = []
    loss_values = []
    for svm_pred, cnn_pred, true_label in zip(svm_predictions, cnn_predictions, y_pred):
        votes = [svm_pred, cnn_pred]
        class_counts = np.bincount(votes)
        majority_vote = np.argmax(class_counts)
        fusion_predictions.append(majority_vote)

        # Cálculo da perda para a previsão da fusão
        true_label_one_hot = np.zeros_like(class_counts)
        true_label_one_hot[true_label] = 1
        loss = categorical_crossentropy([true_label_one_hot], [class_counts])
        loss_values.append(loss)

    # Avaliação do desempenho da fusão das previsões
    accuracy_fusion = accuracy_score(y_pred, fusion_predictions)
    average_loss = np.mean(loss_values)
    print("Acurácia da fusão das previsões: {:.2f}%".format(accuracy * 100))
    print("Perda média da fusão das previsões: {:.4f}".format(average_loss))

    # best_fusion_method = max(results, key=lambda x: results[x]['accuracy'])
    # best_accuracy = results[best_fusion_method]['accuracy']

    # acc_per_fold_fusion.append(best_accuracy)

    # best_fusion_methods_with_best_accuracy = [
    #     method for method in results if results[method]['accuracy'] == best_accuracy]
    # best_fusion_method_with_best_loss = min(
    #     best_fusion_methods_with_best_accuracy, key=lambda x: results[x]['loss'])

    # print('Resultados da fusão:')
    # for fusion_method, metrics in results.items():
    #     print(
    #         f'{fusion_method}: Accuracy = {metrics["accuracy"]}, Loss = {metrics["loss"]}')

    # print(
    #     f'O melhor método de fusão com base na melhor acurácia é {best_fusion_method} (Accuracy = {best_accuracy})')
    # print(
    #     f'O melhor método de fusão com base na melhor acurácia e menor perda é {best_fusion_method_with_best_loss} (Accuracy = {best_accuracy}, Loss = {results[best_fusion_method_with_best_loss]["loss"]})')

    if scores[1] > best_acc:
        best_acc = scores[1]
        best_model = model
    scores_array.append(scores)
    print(
        f'Score for fold (CNN) {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    accuracy.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # cnn_predictions = np.argmax(cnn_predictions, axis=1)
    fused_predictions = np.argmax(fused_predictions, axis=1)

    confusion_matrix_fold = confusion_matrix(
        y_pred, fused_predictions)

    overall_confusion_matrix += confusion_matrix_fold
    plot_confusion_matrix(confusion_matrix_fold, classes=[
        i for i in range(1, 16)], title='Matriz de confusão fold '+str(fold_no)+' fusão CNN - SVM', fold=fold_no)

    plt.figure(figsize=(15, 5))

    plt_loss = plt.subplot(121)
    plt.plot(loss_values, label=f'fold {fold_no}')
    plt.title("Perda")
    plt.ylabel("Perda")
    plt.xlabel("Época")
    plt.legend()

    plt_accuracy = plt.subplot(122)
    plt.plot(accuracy_fusion, label=f'fold {fold_no}')
    plt.title("Acurácia")
    plt.ylabel("Acurácia")
    plt.xlabel("Época")
    plt.legend()

    plt.savefig(f"fusion/cnn_training_results_fold_{fold_no}.png")

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
