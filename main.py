import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import pandas as pd
import tensorflow as tf
import keras
from keras.models import Sequential, load_model
from keras.layers import *
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import keras.backend as K
from keras.backend import get_value, ctc_decode
from keras.optimizers import Nadam
from keras.callbacks import ModelCheckpoint
from keras.utils import CustomObjectScope
import math
def preprocess(img):
    def resize_n_rotate(img, shape_to=(64, 400)):
        if img.shape[0] > shape_to[0] or img.shape[1] > shape_to[1]:
            shrink_multiplier = min(math.floor(shape_to[0] / img.shape[0] * 100) / 100,
                                    math.floor(shape_to[1] / img.shape[1] * 100) / 100)
            img = cv2.resize(img, None, fx=shrink_multiplier, fy=shrink_multiplier, interpolation=cv2.INTER_AREA)
        img = cv2.copyMakeBorder(img, math.ceil(shape_to[0]/2) - math.ceil(img.shape[0]/2),
                                 math.floor(shape_to[0]/2) - math.floor(img.shape[0]/2),
                                 math.ceil(shape_to[1]/2) - math.ceil(img.shape[1]/2),
                                 math.floor(shape_to[1]/2) - math.floor(img.shape[1]/2),
                                 cv2.BORDER_CONSTANT, value=255)
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    def add_adaptiveThreshold(img):
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2).astype('bool')

    for func in [resize_n_rotate, add_adaptiveThreshold]:
        img = func(img)
    return img
alphabet = """ !"%'()+,-./0123456789:;=?R[]abcehinoprstuxy«»АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё№"""

class CERMetric(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the Character Error Rate
    """
    def __init__(self, name='CER_metric', **kwargs):
        super(CERMetric, self).__init__(name=name, **kwargs)
        self.cer_accumulator = self.add_weight(name="total_cer", initializer="zeros")
        self.counter = self.add_weight(name="cer_count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        input_shape = K.shape(y_pred)
        input_length = tf.ones(shape=input_shape[0]) * K.cast(input_shape[1], 'float32')
        decode, log = K.ctc_decode(y_pred, input_length, greedy=True)
        decode = K.ctc_label_dense_to_sparse(decode[0], K.cast(input_length, 'int32'))
        y_true_sparse = K.ctc_label_dense_to_sparse(y_true, K.cast(input_length, 'int32'))
        y_true_sparse = tf.sparse.retain(y_true_sparse, tf.not_equal(y_true_sparse.values,
                                                                     tf.math.reduce_max(y_true_sparse.values)))
        decode = tf.sparse.retain(decode, tf.not_equal(decode.values, -1))
        distance = tf.edit_distance(decode, y_true_sparse, normalize=True)
        self.cer_accumulator.assign_add(tf.reduce_sum(distance))
        self.counter.assign_add(K.cast(len(y_true), 'float32'))

    def result(self):
        return tf.math.divide_no_nan(self.cer_accumulator, self.counter)

    def reset_state(self):
        self.cer_accumulator.assign(0.0)
        self.counter.assign(0.0)

def CTCLoss(y_true, y_pred):
    """
    Compute the training-time loss value
    """
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64") #размер батча
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64") #длины входных данных(предсказания)
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64") #длины выходных данных(метки)

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64") #перевод в тензоры
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length) #сравнивает предсказание (y_pred) с истинным значением (y_true), учитывая длину каждой последовательности
    return loss

with CustomObjectScope({'CTCLoss': CTCLoss, 'CERMetric': CERMetric}):
    model = load_model('ckeckpoint_model.h5')
def num_to_label(num, alphabet):
    text = ""
    for ch in num:
        if ch == len(alphabet):  # ctc blank
            break
        else:
            text += alphabet[ch]
    return text

# Decode labels for softmax matrix
def decode_text(nums):
    values = get_value(
        ctc_decode(nums, input_length=np.ones(nums.shape[0]) * nums.shape[1],
                   greedy=True)[0][0])
    texts = []
    for i in range(nums.shape[0]):
        value = values[i]
        texts.append(num_to_label(value[value >= 0], alphabet))
    return texts

def word_predict(img):
# Преобразование и предобработка изображения

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Применение предобработки к переданному изображению
    preprocessed_img = preprocess(img)
    # Добавление размерностей для модели
    preprocessed_img = np.expand_dims(preprocessed_img, axis=-1)  # Добавление размерности канала
    preprocessed_img = np.expand_dims(preprocessed_img, axis=0)  # Добавление размерности пакета
    # Получение предсказания модели
    prediction = model.predict(preprocessed_img)
    # Декодирование предсказания в текст
    decoded_prediction = decode_text(prediction)
    return decoded_prediction[0]



def clear_folder(folder):
    # Удаляет папку со всем её содержимым и создает её заново
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def text_segment_and_recogn(image):
    # Очистка и создание папки для сохранения изображений предобработки
    save_folder = 'processed_images'
    clear_folder(save_folder)
    colorimg = cv2.imread(image, cv2.IMREAD_COLOR)
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    def bradley_threshold(image):
        height, width = image.shape
        S = width // 8
        s2 = S // 2
        t = 0.15
        res = np.zeros_like(image)
        integral_image = np.zeros((height, width), dtype=np.int64)
        # Рассчитываем интегральное изображение
        for i in range(width):
            sum_ = 0
            for j in range(height):
                sum_ += image[j, i]
                if i == 0:
                    integral_image[j][i] = sum_
                else:
                    integral_image[j][i] = integral_image[j][i - 1] + sum_
        # Находим границы для локальных областей и бинаризуем
        for i in range(width):
            for j in range(height):
                x1 = max(i - s2, 0)
                x2 = min(i + s2, width - 1)
                y1 = max(j - s2, 0)
                y2 = min(j + s2, height - 1)
                count = (x2 - x1) * (y2 - y1)
                sum_ = integral_image[y2][x2] - integral_image[y1][x2] - integral_image[y2][x1] + integral_image[y1][x1]
                if image[j, i] * count < sum_ * (1.0 - t):
                    res[j, i] = 255  # Белый цвет для текста
                else:
                    res[j, i] = 0  # Черный цвет для фона
        return res
    thresh_img = bradley_threshold(img)

    kernel = np.ones((3, 80), np.uint8)
    dilated = cv2.dilate(thresh_img, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours_lines = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

    img_with_lines = img.copy()
    for ctr in sorted_contours_lines:
        x, y, w, h = cv2.boundingRect(ctr)
        cv2.rectangle(img_with_lines, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Сохранение этапов предобработки
    cv2.imwrite(os.path.join(save_folder, '1.original_image.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(save_folder, '2.resized_image.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(save_folder, '3.binary_image.png'), thresh_img)
    cv2.imwrite(os.path.join(save_folder, '4.dilated_lines_image.png'), dilated)
    cv2.imwrite(os.path.join(save_folder, '5.lines_highlighted_image.png'), cv2.cvtColor(img_with_lines, cv2.COLOR_RGB2BGR))


    kernel = np.ones((3, 15), np.uint8)
    dilated_words = cv2.dilate(thresh_img, kernel, iterations=1)

    img_with_words = colorimg .copy()
    recognized_text = []
    for line in sorted_contours_lines:
        line_text = []
        x, y, w, h = cv2.boundingRect(line)
        roi_line = dilated_words[y:y+h, x:x+w]
        contours, _ = cv2.findContours(roi_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sorted_contour_words = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

        for word in sorted_contour_words:
            if cv2.contourArea(word) < 400:
                continue

            x2, y2, w2, h2 = cv2.boundingRect(word)
            cv2.rectangle(img_with_words, (x + x2, y + y2), (x + x2 + w2, y + y2 + h2), (255, 0, 0), 2)
            word_img = img[y + y2:y + y2 + h2, x + x2:x + x2 + w2]
            predicted_word = word_predict(word_img)
            line_text.append(predicted_word)
        if line_text:  # Добавляем строку и перенос только если в ней есть слова
            recognized_text.append(' '.join(line_text))
            recognized_text.append('\n')
    # Сохранение изображений для слов и строк
    cv2.imwrite(os.path.join(save_folder, '6.dilated_words_image.png'), dilated_words)
    cv2.imwrite(os.path.join(save_folder, '7.words_highlighted_image.png'), img_with_words)
    final_text = ''.join(recognized_text)
    print(final_text)
    return final_text

text_segment_and_recogn('TestTexts/SegmentTest18.jpg')