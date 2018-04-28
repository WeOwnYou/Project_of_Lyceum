import pymongo
from PIL import Image
import numpy
from math import e


class NN:
    def __init__(self, width=50, height=50):

        client = pymongo.MongoClient("mongo.yandexlyceum.ru", 27017)
        self.letters_and_coeficients_bd = client.c.Neurons
        self.width = width
        self.height = height


    def add_letter(self, name_letter, RGB_pic_to_load):

        is_letter_in_bd = False
        RGB_arr = []
        img = Image.open(RGB_pic_to_load)
        resized_img = img.resize((self.width, self.height), Image.ANTIALIAS)

        for i in range(self.width):
            RGB_arr.append([])
            for j in range(self.height):
                RGB_arr[i].append(0)

        for i in range(self.width):
            for j in range(self.height):
                RGB_arr[i][j] = resized_img.getpixel((i, j))

        for collection in self.letters_and_coeficients_bd.find():
            try:
                if (collection[name_letter]):
                    is_letter_in_bd = True

            except KeyError:
                is_letter_in_bd = False

        if not (is_letter_in_bd):
            self.letters_and_coeficients_bd.insert_one({name_letter: RGB_arr})


    def learning(self, name_letter, RGB_pic_to_learning):

        is_bd_changed = False
        result_array_of_coeficents_of_bd = []

        for i in range(self.width):
            result_array_of_coeficents_of_bd.append([])
            for j in range(self.height):
                result_array_of_coeficents_of_bd[i].append(0)

        img = Image.open(RGB_pic_to_learning)
        resized_img = img.resize((self.width, self.height), Image.ANTIALIAS)
        resized_img.save("RE.png")
        temporary_coeficients_of_pic = []

        for i in range(self.width):
            temporary_coeficients_of_pic.append([])
            for j in range(self.height):
                temporary_coeficients_of_pic[i].append(0)

        for i in range(self.width):
            for j in range(self.height):
                temporary_coeficients_of_pic[i][j] = resized_img.getpixel((i, j))

        for collection in self.letters_and_coeficients_bd.find():
            try:
                if (collection[name_letter]):
                    temporary_coeficients_of_bd = collection[name_letter]
                    is_bd_changed = True

            except KeyError:
                continue

            for self.width_pixel in range(self.width):
                for self.height_pixel in range(self.height):
                    result_array_of_coeficents_of_bd[self.width_pixel][self.height_pixel] = \
                        (((temporary_coeficients_of_bd[self.width_pixel][self.height_pixel][0]
                           + temporary_coeficients_of_pic[self.width_pixel][self.height_pixel][0]) / 2),
                         ((temporary_coeficients_of_bd[self.width_pixel][self.height_pixel][1]
                           + temporary_coeficients_of_pic[self.width_pixel][self.height_pixel][1]) / 2),
                         ((temporary_coeficients_of_bd[self.width_pixel][self.height_pixel][2]
                           + temporary_coeficients_of_pic[self.width_pixel][self.height_pixel][2]) / 2))

            if (is_bd_changed):
                break

        self.letters_and_coeficients_bd.update({name_letter: temporary_coeficients_of_bd},
                                               {name_letter: result_array_of_coeficents_of_bd})


    def normalization(self, sum, s, normaliztor_for_sum=4, normalizator=50 * 50 * 10, normalizator2=15):

        sum /= normaliztor_for_sum
        s /= normalizator
        s *= normalizator2
        return (sum, s)


    def activation(self, x):

        f = 1 / (1 + e ** (-x))
        return f


    def advanced_predict(self, RGB_pic_to_predict):

        img = Image.open(RGB_pic_to_predict)
        resized_img = img.resize((self.width, self.height), Image.ANTIALIAS)
        resized_img.save("RE.png")
        RGB_arr = []

        for i in range(self.width):
            RGB_arr.append([])
            for j in range(self.height):
                RGB_arr[i].append(0)

        predicted_array = []

        for i in range(self.width):
            for j in range(self.height):
                RGB_arr[i][j] = resized_img.getpixel(((i, j)))

        index = -1

        for collection in self.letters_and_coeficients_bd.find():
            for key in collection:
                if not (key == "_id"):

                    temp_coefs_array = collection[key]
                    sum = 0
                    s = 0
                    const_of_white = 230

                    for i in range(self.width):
                        for j in range(self.height):
                            if not (RGB_arr[i][j][0] >= const_of_white and
                                    RGB_arr[i][j][1] >= const_of_white and RGB_arr[i][j][2] >= const_of_white):
                                a = RGB_arr[i][j][0] + RGB_arr[i][j][1] + RGB_arr[i][j][2]
                                b = temp_coefs_array[i][j][0] + temp_coefs_array[i][j][1] + temp_coefs_array[i][j][2]
                                if (abs(a - b) < 100):
                                    sum += 1
                                    s += abs(a - b)

                    sum, x = self.normalization(sum, s, 4, self.width * self.height * 10, 15)
                    result = self.activation(x)
                    if (x == 0):
                        x = sum
                        result = self.activation(x)

                    try:
                        for index in range(len(predicted_array)):
                            temp = int(predicted_array[index][0] * 100)
                            if (int(result * 100) == temp):
                                if (int(predicted_array[index][2]) > sum):
                                    result -= 0.2
                                else:
                                    predicted_array[index][0] -= 0.2
                    except IndexError:
                        pass

                    predicted_array.append([result, key, sum])
                    index += 1

        is_sum_is_main = False
        max = -1
        ind_max = 0
        letter_ind = ""
        max_O = 0

        for i in range(len(predicted_array)):
            if (predicted_array[i][1] == 'O'):
                max_O = predicted_array[i][2]
            if (predicted_array[i][2] > max):
                max = predicted_array[i][2]
                ind_max = i
                letter_ind = predicted_array[i][1]

        for i in range(len(predicted_array)):
            if (ind_max == i):
                continue
            if (max > predicted_array[i][2] + 5.9):
                is_sum_is_main = True
            else:
                is_sum_is_main = False
                break

        if (not is_sum_is_main and (letter_ind == "Q" and abs(max_O - max) < 5)):
            return "O"
        elif (not (letter_ind == "Q" and abs(max_O - max) < 5)):
            return predicted_array[ind_max][1]

        for i in range(len(predicted_array)):
            for j in range(len(predicted_array) - i - 1):
                if (predicted_array[j][0] < predicted_array[j + 1][0]):
                    temp = predicted_array[j]
                    predicted_array[j] = predicted_array[j + 1]
                    predicted_array[j + 1] = temp

        if (predicted_array[0][2] * 2 < predicted_array[1][2]):
            return predicted_array[1][1]
        return predicted_array[0][1]



    def live_pic_to_white(self, RGB_pic):

        img = Image.open(RGB_pic)
        resized_img = img.resize((self.width, self.height), Image.ANTIALIAS)
        resized_img.save("RE.png")
        RGB_arr = []

        for i in range(self.width):
            RGB_arr.append([])
            for j in range(self.height):
                RGB_arr[i].append(0)

        const_of_white = 110

        for i in range(self.width):
            for j in range(self.height):
                RGB_arr[i][j] = resized_img.getpixel((i, j))
                if (RGB_arr[i][j][0] > const_of_white or RGB_arr[i][j][1] > const_of_white or RGB_arr[i][j][
                    2] > const_of_white):
                    RGB_arr[i][j] = (255, 255, 255)
        return RGB_arr



    def live_predic(self, RGB_pic):

        arr = self.live_pic_to_white(RGB_pic)
        predicted_array = []
        index = -1

        for collection in self.letters_and_coeficients_bd.find():
            for key in collection:
                if not (key == "_id"):
                    temp_coefs_array = collection[key]
                    sum = 0
                    s = 0

                    for i in range(self.width):
                        for j in range(self.height):
                            if not (arr[i][j][0] >= 230 and arr[i][j][1] >= 230 and arr[i][j][2] >= 230):
                                a = arr[i][j][0] + arr[i][j][1] + arr[i][j][2]
                                b = temp_coefs_array[i][j][0] + temp_coefs_array[i][j][1] + temp_coefs_array[i][j][2]
                                if (abs(a - b) < 100):
                                    sum += 1
                                    s += abs(a - b)

                    sum, x = self.normalization(sum, s, 4, self.width * self.height * 10, 15)
                    result = self.activation(x)
                    if (x == 0):
                        x = sum
                        result = self.activation(x)

                    try:
                        for index in range(len(predicted_array)):
                            temp = int(predicted_array[index][0] * 100)
                            if (int(result * 100) == temp):
                                if (int(predicted_array[index][2]) > sum):
                                    result -= 0.2
                                else:
                                    predicted_array[index][0] -= 0.2
                    except IndexError:
                        pass

                    predicted_array.append([result, key, sum])
                    index += 1

        is_sum_is_main = False
        max = -1
        ind_max = 0
        letter_ind = ""

        for i in range(len(predicted_array)):
            if (predicted_array[i][2] > max):
                max = predicted_array[i][2]
                ind_max = i
                letter_ind = predicted_array[i][1]

        for i in range(len(predicted_array)):
            if (ind_max == i):
                continue
            if (max > predicted_array[i][2] + 5.9):  # 9.5
                is_sum_is_main = True
            else:
                is_sum_is_main = False
                break

        if (is_sum_is_main or letter_ind == "K" or letter_ind == "Q"):
            return predicted_array[ind_max][1]

        for i in range(len(predicted_array)):
            for j in range(len(predicted_array) - i - 1):
                if (predicted_array[j][0] < predicted_array[j + 1][0]):
                    temp = predicted_array[j]
                    predicted_array[j] = predicted_array[j + 1]
                    predicted_array[j + 1] = temp

        if (predicted_array[0][2] * 2 < predicted_array[1][2]):
            return predicted_array[1][1]

        return predicted_array[0][1]
        
        
        from NeuronNerwork import *
NeuronNetwork = NN()
# a.add_letter("Z", "result1.png")
# a.learning("Z","result1.png")
print(NeuronNetwork.advanced_predict("res1.png"))
