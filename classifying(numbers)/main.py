import os #для работы с ос
import random

import cv2 # для подгрузки изображения
import numpy as np

from tqdm import tqdm

def data_loader(path): #хреновый дата лоудер , так как грузит в оперативу дата сет полностью
    data = [] #запихиваем сюда все изображения
    for index in range(10): #проходимся по классам нашим
        clas = np.int16(index) #конвентируем индекс в число нампая
        path_clas = os.path.join(path, f"class_{index}")
        for filename in tqdm(os.listdir(path_clas)): #
            path_file = os.path.join(path_clas, filename) #
            imeig = cv2.imread(path_file, cv2.IMREAD_GRAYSCALE) #путь к файлу запихиваем сюда
            imeig = np.resize(imeig, (1, 28 * 28))/255 #это матрица входа в которую мы всовываем изображение, растянули картинку в линию
            data.append((imeig, clas)) #добавляем дату в картеж
    return data
print ("загрузка")
data_training = data_loader("mnist/training/")
data_testing = data_loader("mnist/testing")



# математические функции
def relu(t):
    return np.maximum(t, 0)



def relu(t):
    return np.maximum(t, 0)


def softmax(t):
    out = np.exp(t)
    return out / np.sum(out)


def softmax_batch(t):
    out = np.exp(t)
    return out / np.sum(out, axis=1, keepdims=True)


def sparse_cross_entropy(z, y):
    return -np.log(z[0, y])


def sparse_cross_entropy_batch(z, y):
    return -np.log(np.array([z[j, y[j]] for j in range(len(y))]))


def to_full(y, num_classes):
    y_full = np.zeros((1, num_classes))
    y_full[0, y] = 1
    return y_full


def to_full_batch(y, num_classes):
    y_full = np.zeros((len(y), num_classes))
    for j, yj in enumerate(y):
        y_full[j, yj] = 1
    return y_full


def relu_deriv(t):
    return (t >= 0).astype(float)



INPUT_DIM = 784 #колличество входных значений 4 признака
OUT_DIM = 10 #колличество выходных значений 3, по тому что 3 класса
H_DIM = 128 #колличество нейронов в первом слое, ибо просто так






W1 = np.random.rand(INPUT_DIM, H_DIM)
b1 = np.random.rand(1, H_DIM)
W2 = np.random.rand(H_DIM, OUT_DIM)
b2 = np.random.rand(1, OUT_DIM)
#%%
W1 = (W1 - 0.5) * 2 * np.sqrt(1 / INPUT_DIM)
b1 = (b1 - 0.5) * 2 * np.sqrt(1 / INPUT_DIM)
W2 = (W2 - 0.5) * 2 * np.sqrt(1 / H_DIM)
b2 = (b2 - 0.5) * 2 * np.sqrt(1 / H_DIM)




#блок с обучением нейронки
ALPHA = 0.0002
NUM_EPOCHS = 10
BATCH_SIZE = 8

loss_arr = []

for ep in range(NUM_EPOCHS):
    random.shuffle(data_training)
    for i in tqdm(range(len(data_training) // BATCH_SIZE), desc=f"EPOHA: {ep}"): #tqdm делает красоту в иде с загрузочкой
        batch_x, batch_y = zip(*data_training[i * BATCH_SIZE: i * BATCH_SIZE + BATCH_SIZE])
        x = np.concatenate(batch_x, axis=0)
        y = np.array(batch_y)

        # Forward
        t1 = x @ W1 + b1
        h1 = relu(t1)
        t2 = h1 @ W2 + b2
        z = softmax_batch(t2)

        #блок который специально вызывает ошибку чтобы прекратить обучение в случае поломке нейронки
        if np.isnan(z.min()):
            print("EPOHA: ", ep, "BATH: ", i)
            print(1/0)


        E = np.sum(sparse_cross_entropy_batch(z, y))

        # Backward
        y_full = to_full_batch(y, OUT_DIM)
        dE_dt2 = z - y_full
        dE_dW2 = h1.T @ dE_dt2
        dE_db2 = np.sum(dE_dt2, axis=0, keepdims=True)
        dE_dh1 = dE_dt2 @ W2.T
        dE_dt1 = dE_dh1 * relu_deriv(t1)
        dE_dW1 = x.T @ dE_dt1
        dE_db1 = np.sum(dE_dt1, axis=0, keepdims=True)

        # Update
        W1 = W1 - ALPHA * dE_dW1
        b1 = b1 - ALPHA * dE_db1
        W2 = W2 - ALPHA * dE_dW2
        b2 = b2 - ALPHA * dE_db2

        loss_arr.append(E)

def predict(x):
    t1 = x @ W1 + b1  # @ это способ умножения матрицы на вектор или на матрицу
    h1 = relu(t1)
    t2 = h1 @ W2 + b2
    z = softmax(t2)
    return z



#тест на сколько хорошо нейросеть проходит данные
v=0
for data in data_testing:
    res,oz = np.argmax(predict(data[0])), data[1]
    if res==oz:
        v += 1
print(len(data_testing),v)