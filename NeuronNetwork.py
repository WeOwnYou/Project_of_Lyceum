import pymongo
from PIL import Image
import numpy
from math import e
def Sort(array):
    return array[0]

class NN:
    def __init__(self):
        client = pymongo.MongoClient("mongo.yandexlyceum.ru", 27017)
        self.letters_and_coeficients_bd = client.b.Neurons
        # self.letters_and_coeficients_bd.remove()
        for collection in self.letters_and_coeficients_bd.find():
            # try:
            #     self.letters_and_coeficients_bd.remove({"В": collection["В"]})
            # except Exception:
            #     pass
            print(collection)
        print(self.letters_and_coeficients_bd.count())


    def add_letter(self, name_letter, RGB_pic):
        width = 50
        height = 50
        is_letter_in_bd = False
        img = Image.open(RGB_pic)
        resized_img = img.resize((width, height), Image.ANTIALIAS)

        arr = []
        for i in range(width):
            arr.append([])
            for j in range(height):
                arr[i].append(0)


        for i in range(width):
            for j in range(height):
                arr[i][j] = resized_img.getpixel((i, j))
                print(arr[i][j])

        for collection in self.letters_and_coeficients_bd.find():
            try:
                if(collection[name_letter]):
                    is_letter_in_bd = True
            except Exception:
                is_letter_in_bd = False
        if not(is_letter_in_bd):
            self.letters_and_coeficients_bd.insert_one({name_letter: arr})
            
            
    def learning(self, name_letter ,RGB_pic_to_learning):
        width = 50
        height = 50
        is_bd_changed = False

        result_array_of_coeficents_of_bd = []
        for i in range(width):
            result_array_of_coeficents_of_bd.append([])
            for j in range(height):
                result_array_of_coeficents_of_bd[i].append(0)
        # result = RGB_pic_to_learning
        img = Image.open(RGB_pic_to_learning)
        resized_img = img.resize((width, height), Image.ANTIALIAS)
        resized_img.save("resize.png")
        temporary_coeficients_of_pic = []
        for i in range(width):
            temporary_coeficients_of_pic.append([])
            for j in range(height):
                temporary_coeficients_of_pic[i].append(0)


        for i in range(width):
            for j in range(height):
                temporary_coeficients_of_pic[i][j] =  resized_img.getpixel((i, j))
                print(temporary_coeficients_of_pic[i][j])
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

            for width_pixel in range(width):
                for height_pixel in range(height):
                    result_array_of_coeficents_of_bd[width_pixel][height_pixel] = \
                                                        (((temporary_coeficients_of_bd[width_pixel][height_pixel][0]+temporary_coeficients_of_pic[width_pixel][height_pixel][0]) / 2),
                                                        ((temporary_coeficients_of_bd[width_pixel][height_pixel][1] + temporary_coeficients_of_pic[width_pixel][height_pixel][1]) / 2),
                                                        ((temporary_coeficients_of_bd[width_pixel][height_pixel][2] + temporary_coeficients_of_pic[width_pixel][height_pixel][2]) / 2))
            # print(result_array_of_coeficents_of_bd)
            if(is_bd_changed):
                break

        self.letters_and_coeficients_bd.update({name_letter: temporary_coeficients_of_bd},
                                               {name_letter: result_array_of_coeficents_of_bd})
    def predict(self, right_letter, RGB_pic_to_predict):
        width = 50
        height = 50
        img = Image.open(RGB_pic_to_predict)
        resized_img = img.resize((width, height), Image.ANTIALIAS)
        resized_img.save("resize.png")
        arr = []
        for i in range(width):
            arr.append([])
            for j in range(height):
                arr[i].append(0)
        # print(arr)
        for i in range(width):
            for j in range(height):
                arr[i][j] = resized_img.getpixel(((i, j)))
                # print(arr[i][j])
        # print(arr)
        s = 0
        sum = 0
        for collection in self.letters_and_coeficients_bd.find():
            try:
                temp_coefs_array = collection[right_letter]
                # print(temp_coefs_array)
                for i in range(width):
                    for j in range(height):
                        print(arr[i][j], temp_coefs_array[i][j],sep = "\t\t")
                        # print(not(arr[i][j][0] == 255 and arr[i][j][1] == 255 and arr[i][j][2] == 255), end = "\n")
                        # print(not (temp_coefs_array[i][j][0] == 255 and temp_coefs_array[i][j][1] == 255 and temp_coefs_array[i][j][2] == 255))
                        # print(abs(arr[i][j][0]-temp_coefs_array[i][j][0])<100)
                        # print(abs(arr[i][j][1]-temp_coefs_array[i][j][1])<100)
                        # print(abs(arr[i][j][2]-temp_coefs_array[i][j][2]) < 100)
                        if not(arr[i][j][0] >= 230 and arr[i][j][1] >= 230 and arr[i][j][2] >= 230): #\
                                # and abs(arr[i][j][0]-temp_coefs_array[i][j][0])< 100 and abs(arr[i][j][1]-temp_coefs_array[i][j][1])<100 \
                                # and abs(arr[i][j][2]-temp_coefs_array[i][j][2]) < 100:
                                # and not(temp_coefs_array[i][j][0] == 255.0 and temp_coefs_array[i][j][1] == 255.0 and temp_coefs_array[i][j][2] == 255.0 )\
                            # print(arr[i][j], temp_coefs_array[i][j], sep = "\t\t")
                            a = arr[i][j][0] + arr[i][j][1] + arr[i][j][2]
                            b = temp_coefs_array[i][j][0] + temp_coefs_array[i][j][1] + temp_coefs_array[i][j][2]
                            if(abs(a-b)<100):
                                sum+=1
                                print(arr[i][j], temp_coefs_array[i][j], sep = "\t\t")
                                s+=abs(a-b)
            except KeyError:
                print(Exception)
                continue
        sum/=4
        x = (s/(width*height*10)*sum)
        print(sum, x)
        result = 1 / (1 + e ** (-x))
        print(result)
    def advanced_predict(self,RGB_pic_to_predict):
        width = 50
        height = 50
        img = Image.open(RGB_pic_to_predict)
        resized_img = img.resize((width, height), Image.ANTIALIAS)
        resized_img.save("resize.png")
        arr = []
        for i in range(width):
            arr.append([])
            for j in range(height):
                arr[i].append(0)
        predicted_array = []
        for i in range(width):
            for j in range(height):
                arr[i][j] = resized_img.getpixel(((i, j)))
        index =  -1
        for collection in self.letters_and_coeficients_bd.find():
            for key in collection:
                if not(key == "_id"):
                    temp_coefs_array = collection[key]
                    sum = 0
                    s = 0
                    for i in range(width):
                        for j in range(height):
                            if not (arr[i][j][0] >= 230 and arr[i][j][1] >= 230 and arr[i][j][2] >= 230):
                                a = arr[i][j][0] + arr[i][j][1] + arr[i][j][2]
                                b = temp_coefs_array[i][j][0] + temp_coefs_array[i][j][1] + temp_coefs_array[i][j][2]
                                if (abs(a - b) < 100):
                                    sum += 1
                                    # print(arr[i][j], temp_coefs_array[i][j], sep="\t\t")
                                    s += abs(a - b)
                    sum /= 4
                    x = (s / (width * height * 10)*15)
                    print(sum, x, key)
                    result = 1 / (1 + e ** (-x))
                    if(x == 0):
                        x = sum
                        result = 1 / (1 + e ** (-x))
                        # print(result)
                    print(result, predicted_array)
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
                    index += 1
        for i in range(len(predicted_array)):
            for j in range(len(predicted_array) - i - 1):
                if (predicted_array[j][0] < predicted_array[j + 1][0]):
                    temp = predicted_array[j]
                    predicted_array[j] = predicted_array[j + 1]
                    predicted_array[j + 1] = temp
        print(predicted_array)
        return predicted_array[0][1]

a = NN()
# a.add_letter("В", "В.png")
# a.learning("Б","Б6.png")
# a.add_letter("А", "А3.png")
# a.learning("А", "А1.png")
print(a.advanced_predict("Аlive1.png"))
