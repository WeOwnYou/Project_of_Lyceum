import pymongo
from PIL import Image
import numpy
from math import e

class NN:
    def __init__(self):
        client = pymongo.MongoClient("mongo.yandexlyceum.ru", 27017)
        self.letters_and_coeficients_bd = client.c.Neurons
        self.width = 50
        self.height = 50
        for collection in self.letters_and_coeficients_bd.find():
            try:
               self.letters_and_coeficients_bd.remove({"C": collection["C"]})
            except Exception:
                pass
            print(collection)
            pass
        print(self.letters_and_coeficients_bd.count())


    def add_letter(self, name_letter, RGB_pic):
        is_letter_in_bd = False
        img = Image.open(RGB_pic)
        resized_img = img.resize((self.width, self.height), Image.ANTIALIAS)

        arr = []
        for i in range(self.width):
            arr.append([])
            for j in range(self.height):
                arr[i].append(0)


        for i in range(self.width):
            for j in range(self.height):
                arr[i][j] = resized_img.getpixel((i, j))
                # print(arr[i][j])

        for collection in self.letters_and_coeficients_bd.find():
            try:
                if(collection[name_letter]):
                    is_letter_in_bd = True
            except Exception:
                is_letter_in_bd = False
        if not(is_letter_in_bd):
            self.letters_and_coeficients_bd.insert_one({name_letter: arr})
            
            
    def learning(self, name_letter ,RGB_pic_to_learning):
        self.width = 50
        self.height = 50
        is_bd_changed = False

        result_array_of_coeficents_of_bd = []
        for i in range(self.width):
            result_array_of_coeficents_of_bd.append([])
            for j in range(self.height):
                result_array_of_coeficents_of_bd[i].append(0)
        # result = RGB_pic_to_learning
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
                temporary_coeficients_of_pic[i][j] =  resized_img.getpixel((i, j))
                # print(temporary_coeficients_of_pic[i][j])
        # print(temporary_coeficients_of_pic)


        for collection in self.letters_and_coeficients_bd.find():
            try:
                if(collection[name_letter]):
                    temporary_coeficients_of_bd = collection[name_letter]
                    # print(collection[name_letter],"!")
                    is_bd_changed = True

            except Exception:
                # print(Exception)
                continue

            # print(temporary_coeficients_of_bd)

            for self.width_pixel in range(self.width):
                for self.height_pixel in range(self.height):
                    result_array_of_coeficents_of_bd[self.width_pixel][self.height_pixel] = \
                                                        (((temporary_coeficients_of_bd[self.width_pixel][self.height_pixel][0]+temporary_coeficients_of_pic[self.width_pixel][self.height_pixel][0]) / 2),
                                                        ((temporary_coeficients_of_bd[self.width_pixel][self.height_pixel][1] + temporary_coeficients_of_pic[self.width_pixel][self.height_pixel][1]) / 2),
                                                        ((temporary_coeficients_of_bd[self.width_pixel][self.height_pixel][2] + temporary_coeficients_of_pic[self.width_pixel][self.height_pixel][2]) / 2))
            # print(result_array_of_coeficents_of_bd)
            if(is_bd_changed):
                break

        self.letters_and_coeficients_bd.update({name_letter: temporary_coeficients_of_bd},
                                               {name_letter: result_array_of_coeficents_of_bd})
    def advanced_predict(self,RGB_pic_to_predict):
        self.width = 50
        self.height = 50
        img = Image.open(RGB_pic_to_predict)
        resized_img = img.resize((self.width, self.height), Image.ANTIALIAS)
        resized_img.save("RE.png")
        arr = []
        for i in range(self.width):
            arr.append([])
            for j in range(self.height):
                arr[i].append(0)
        predicted_array = []
        for i in range(self.width):
            for j in range(self.height):
                arr[i][j] = resized_img.getpixel(((i, j)))
                # print(arr[i][j])
        index =  -1
        for collection in self.letters_and_coeficients_bd.find():
            for key in collection:
                if not(key == "_id"):
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
                                    # print(arr[i][j], temp_coefs_array[i][j], sep="\t\t")
                                    s += abs(a - b)
                    sum /= 4
                    x = (s / (self.width * self.height * 10)*15)
                    # print(sum, x, key)
                    result = 1 / (1 + e ** (-x))
                    if(x == 0):
                        x = sum
                        result = 1 / (1 + e ** (-x))
                        # print(result)
                    # print(result, predicted_array)
                    try:
                        for index in range(len(predicted_array)):
                            temp = int(predicted_array[index][0]*100)
                            # print(int(result*100), temp)
                            if(int(result*100) == temp):
                                if(int(predicted_array[index][2])>sum):
                                    result-=0.2
                                else:
                                    predicted_array[index][0]-=0.2
                    except IndexError:
                        pass
                    # print(key, result)
                    # predicted_array.append([result,key])
                    predicted_array.append([result, key, sum])
                    # print(predicted_array)
                    index += 1
        is_sum_is_main = False
        max = -1
        ind_max = 0
        letter_ind = ""
        max_O = 0
        for i in range(len(predicted_array)):
            if(predicted_array[i][1] == 'O'):
                max_O = predicted_array[i][2]
            if(predicted_array[i][2] > max):
                max = predicted_array[i][2]
                ind_max = i
                letter_ind = predicted_array[i][1]
        for i in range(len(predicted_array)):
            if(ind_max == i):
                continue
            if(max > predicted_array[i][2] + 5.9):#9.5
                is_sum_is_main = True
            else:
                is_sum_is_main = False
                break
        if(not is_sum_is_main  and (letter_ind == "Q" and abs(max_O-max)<5)):
            print(predicted_array)
            return "O"
            # print("Da")
        elif(not(letter_ind == "Q" and abs(max_O-max)<5)):
            print(predicted_array)
            return predicted_array[ind_max][1]
        for i in range(len(predicted_array)):
            for j in range(len(predicted_array) - i - 1):
                if (predicted_array[j][0] < predicted_array[j + 1][0]):
                    # print(predicted_array[j], predicted_array[j+1], sep ="\t\t")
                    temp = predicted_array[j]
                    predicted_array[j] = predicted_array[j + 1]
                    predicted_array[j + 1] = temp
        print(predicted_array)
        if(predicted_array[0][2]*2 < predicted_array[1][2]):
            return predicted_array[1][1]
        return predicted_array[0][1]

    def live_to_white(self,RGB_pic):
        self.width = 50
        self.height = 50
        img = Image.open(RGB_pic)
        print(RGB_pic)
        resized_img = img.resize((self.width, self.height), Image.ANTIALIAS)
        resized_img.save("RE.png")

        arr = []
        for i in range(self.width):
            arr.append([])
            for j in range(self.height):
                arr[i].append(0)

        for i in range(self.width):
            for j in range(self.height):
                arr[i][j] = resized_img.getpixel((i, j))
                if(arr[i][j][0] > 110 or arr[i][j][1] > 110 or arr[i][j][2] > 110):
                    arr[i][j]=(255,255,255)
        return arr
    def live_predic(self, RGB_pic):
        self.width = 50
        self.height = 50
        arr = self.live_to_white(RGB_pic)
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
                                    # print(arr[i][j], temp_coefs_array[i][j], sep="\t\t")
                                    s += abs(a - b)
                    sum /= 4
                    x = (s / (self.width * self.height * 10) * 15)
                    # print(sum, x, key)
                    result = 1 / (1 + e ** (-x))
                    if (x == 0):
                        x = sum
                        result = 1 / (1 + e ** (-x))
                        # print(result)
                    # print(result, predicted_array)
                    try:
                        for index in range(len(predicted_array)):
                            temp = int(predicted_array[index][0] * 100)
                            # print(int(result*100), temp)
                            if (int(result * 100) == temp):
                                if (int(predicted_array[index][2]) > sum):
                                    result -= 0.2
                                else:
                                    predicted_array[index][0] -= 0.2
                    except IndexError:
                        pass
                    # print(key, result)
                    # predicted_array.append([result,key])
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
            # print("Da")
            return predicted_array[ind_max][1]
        for i in range(len(predicted_array)):
            for j in range(len(predicted_array) - i - 1):
                #         if (is_sum_is_main):
                #             break
                if (predicted_array[j][0] < predicted_array[j + 1][0]):
                    # print(predicted_array[j], predicted_array[j+1], sep ="\t\t")
                    temp = predicted_array[j]
                    predicted_array[j] = predicted_array[j + 1]
                    predicted_array[j + 1] = temp
        print(predicted_array)
        if (predicted_array[0][2] * 2 < predicted_array[1][2]):
            return predicted_array[1][1]
        return predicted_array[0][1]
a = NN()
# a.add_letter("Z", "result1.png")
# a.learning("Z","result1.png")
# for i in range(3):
#     a.learning("S", "res1.png")
# print(a.advanced_l      ive_predict("Аlivefinale.png"))
# print(a.advanced_live_predict("Бfinale1.png"))
# print(a.advanced_live_predict("Бfinale2.png"))
# print(a.advanced_predict("Бfinale.png"))
# print(a.advanced_predict("res1.png")) #сделать сортировку в лайве по сумме
# print(a.advanced_predict("Elive.png"))
# print(a.live_predic("Elive.png"))
# print(a.advanced_predict(""))
