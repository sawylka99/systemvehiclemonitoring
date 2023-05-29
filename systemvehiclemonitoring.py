import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from math import *
from PIL import Image
from matplotlib import cm
from torchvision import transforms
from intersection import intersection
from scipy.signal import savgol_filter
from shapely.geometry import Point, Polygon
from shapely.geometry import  LineString



utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
      
    
def usedBox (secondFrame, usedBoxes):
    a = secondFrame   
    b = usedBoxes 
    res = [obj for obj in a if obj not in b] + [obj for obj in b if obj not in a]
    if res == []:
        return True
    else:
        return False

def notUsedBox (secondFrame, usedBoxes):
    a = secondFrame   
    b = usedBoxes 
    res = [obj for obj in a if obj not in b] + [obj for obj in b if obj not in a]
    return res

class ObjectDetectionPipeline:
    def __init__(self, threshold=0.5, device="cpu", cmap_name="tab10_r"):
        # First we need a Transform object to turn numpy arrays to normalised tensors.
        # We are using an SSD300 model that requires 300x300 images.
        # The normalisation values are standard for pretrained pytorch models.
        # self.exit_mask = exit_mask
        self.tfms = transforms.Compose([
            transforms.Resize(300),
            transforms.CenterCrop(300),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Next we need a model. We're setting it to evaluation mode and sending it to the correct device.
        # We get some speedup from the gpu but not as much as we could.
        # A more efficient way to do this would be to collect frames to a buffer,
        # run them through the network as a batch, then output them one by one
        self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd').eval().to(device)
        
        # Stop the network from keeping gradients.
        # It's not required but it gives some speedup / reduces memory use.
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.device = device

        self.threshold = threshold  # Confidence threshold for displaying boxes.
        self.cmap = cm.get_cmap(cmap_name)  # colour map
        self.classes_to_labels = utils.get_coco_object_dictionary()
        print(self.classes_to_labels)
        print(len(self.classes_to_labels))
        exit()
        self.deleteBoxmass = [] # Массив машинок под удаление
        self.unUsedboxes = []   # Массив машинок не нашедших совпадений
        self.usedBoxes = []     # Массив машинок нашедших совпадение
        self.newBoxesfdict = [] # Массив для создания новых машинок
        self.firstFrame = []    # Массив машинок на первом кадре или при пустом словаре
        self.secondFrame = []   # Массив машинок на втором или n кадре
        self.dictCoordinate = {}# Словарик {id : [1,2,3,4] - id = countCars, ключ x,y,x1,y1 (x,y- Левый верхний угол бокса, x1,y1- Правый нижний) }
        self.dictCenter = {}    # Словарь траектории (Центры авто)


        self.center_point = None# Точка центра бокса машинки
        self.currentBox = None  # Переменная для нахождения точного совпадение бокса и последующего складывания в словарик 
        self.minDistance = None # Переменная минимального расстояния между центрами машинок для нахождения минимального


        self.countCars = 0      # id машинок
        self.counterDict = 0    
        self.counterDictin = 1


        self.usedBoxesflag = False # Флаг для сравнения использованных машинок и машинок во втором кадре
        self.minDistanceFlag = False# Флаг для проверки есть ли уже совпадение и минимальное расстояние между ними

        self.coords = Polygon # Полигон в котором происходит обнаружение машин
        self.Zone1 = []   # Полигон выездной зоны 1

        # self.Line = Lines   # Массив выездных линий
        self.lineZone1 = [] # Массив для выездной линии 1
        self.originalShape = [640, 480]
        self.resizedShape = [960, 960]


    @staticmethod
    def _crop_img(img):
        scale_percent = 60  # percent of original size
        width = 960  # int(img.shape[1] * scale_percent / 100)
        height = 960  # int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        """Crop an image or batch of images to square"""
        if len(img.shape) == 3:
            y = img.shape[0]
            x = img.shape[1]
        elif len(img.shape) == 4:
            y = img.shape[1]
            x = img.shape[2]
        else:
            raise ValueError(f"Image shape: {img.shape} invalid")

        out_size = min((y, x))
        startx = x // 2 - out_size // 2
        starty = y // 2 - out_size // 2

        if len(img.shape) == 3:
            return img[starty:starty + out_size, startx:startx + out_size]
        elif len(img.shape) == 4:
            return img[:, starty:starty + out_size, startx:startx + out_size]

    def getCenter(self, a):
        """ Функция для нахождения центра машинки из массива бокса(Правого верхнего угла и левого нижнего)
        >>>getCenter([1, 2, 3, 4])
        (x, y)
        """
        x1, y1, x2, y2 = a[0], a[1], a[2], a[3]
        
        w_boxes = x2 - x1
        h_boxes = y2 - y1

        x_d = int(w_boxes / 2)
        y_d = int(h_boxes / 2)

        cx = x1 + x_d
        cy = y1 + y_d

        return (cx, cy)

    def distanceBetweenC(self, fistBox, secondBox):
        """ Функция для нахождения расстояния между центрами боксов
        >>>distanceBetweenC([10, 20, 30, 40], [4, 3, 2, 1])
        (41)
        """
        firstCx = fistBox[0] + fistBox[2] / 2
        firstCy = fistBox[1] + fistBox[3] / 2

        secondCx = secondBox[0] + secondBox[2] / 2
        secondCy = secondBox[1] + secondBox[3] / 2

        return int(sqrt((secondCx - firstCx) ** 2 + (secondCy - firstCy) ** 2))

    def getCentroid(self, x1, y1, x2, y2):
        """ Функция для нахождения центра бокса, его высоты и ширины
        >>>getCentroid(10, 20, 30, 40)
        (cx, cy), w_boxes, h_boxes
        """
        w_boxes = x2 - x1
        h_boxes = y2 - y1

        x_d = int(w_boxes / 2)
        y_d = int(h_boxes / 2)

        cx = x1 + x_d
        cy = y1 + y_d

        return (cx, cy), w_boxes, h_boxes
    
    def getCentr(self, mass):
        """ Функция для нахождения центра бокса, его высоты и ширины
        >>>getCentroid(10, 20, 30, 40)
        (cx, cy), w_boxes, h_boxes
        """
        x1, y1, x2, y2 = mass
        w_boxes = x2 - x1
        h_boxes = y2 - y1

        x_d = int(w_boxes / 2)
        y_d = int(h_boxes / 2)

        cx = x1 + x_d
        cy = y1 + y_d

        return cx, cy


    def optimizeTraker(self, massOptimize):
        if len(massOptimize) <= 8:
            return massOptimize
        X1 = []
        Y1 = []
        for x in massOptimize:
            X1.append(x[0])
            Y1.append(x[1])
        lenWindow = len(X1)/5
        if lenWindow % 2 == 0 and lenWindow >4:
            lenWindow = lenWindow - 1
        else:
            lenWindow = lenWindow + 1
        xFiltred = savgol_filter(X1, lenWindow, 1, mode="nearest")
        massOptimize.clear()

        for x,y in zip(xFiltred, Y1):
            massOptimize.append((int(x),int(y)))
        return massOptimize

    def chechIntersection(self, massCheck):

        # line1 = LineString([line1])
        # line2 = LineString([line2])
        # line3 = LineString([line3])
        line1 = LineString([(455,420) ,(270,960)])
        line2 = LineString([(498,420) ,(482,960)])
        line3 = LineString([(531,420) ,(695,960)])
        colorCar ={}
        newDictCheck ={}
        for key, value in massCheck.items():
            if len(value) >=2:
                newDictCheck[key] = value
            else:
                colorCar[key] = (110,252,69)
        for key, value in newDictCheck.items():
            countIntersection = 0
            path =  LineString(value)
                
            if not path.intersection(line1).is_empty:
                countIntersection+= 1
            if not path.intersection(line2).is_empty:
                countIntersection+= 1
            if not path.intersection(line3).is_empty:
                countIntersection+= 1


            if countIntersection == 0:
                colorCar[key] = (69, 252, 110)
            if countIntersection == 1:
                colorCar[key] = (69, 251, 252)
            if countIntersection >= 2:
                colorCar[key] = (0,13,230)
        
        return colorCar
    
    def clearDictcoordinate(self, delDict, img):
        """ Функция для отчищения словарика массивом ключей, которые необходимо удалить
        >>>clearDictcoordinate([1, 3])
        """
        dictLastTraker = {}
        
        for key, value in self.dictCenter.items():
            if key in delDict:
                dictLastTraker[key] = self.optimizeTraker(value)
        
        colorCarCheck = self.chechIntersection(dictLastTraker)
        self.drowTracker(img, dictLastTraker, colorCarCheck)

        for key in delDict:
            del self.dictCoordinate[key]
            del self.dictCenter[key]


    def drowTracker(self, img, dictCenter, dictColor):
        for id, x in dictCenter.items():
            points = np.array(x)
            points = points.reshape((-1, 1, 2))
            if bool(dictColor):
                cv2.polylines(img, [points], False, dictColor[id], 2)

                startPoint = (self.dictCoordinate[id][-1][0], self.dictCoordinate[id][-1][1])
                endPoint = (self.dictCoordinate[id][-1][2], self.dictCoordinate[id][-1][3])
                cv2.rectangle(img, startPoint, endPoint, dictColor[id], 2)  
           

    def matchesDictcoordinate (self, deleteBoxmass, img):
        """ Функция для нахождения машинки, попавшей в выездную зону,
        проверки пересечения луча по ее последней точки и точки попавшей в выездную зону
        с линиией выезда, записыванием ее id, при находении совпадения, функция удаляет 
        данную машинку из словарика и возвращает True, иначе ничего не происходит
        >>>matchesDictcoordinate([1, 2, 3, 4], [(630, 175), (5100, 210)])
        True
        """
        sizeDict = len(self.dictCoordinate)
        countDict = 1
        delDict = []
        ray = []
        minDistance = 80
        minDistanceFlag = False
        currentCar = 0
        currentFrame = []
        '''
        Принцип работы цикла

        Из словарика берется его последний ключ
        (коодринаты бокса машинки левого верхнего и правого нижнего угла)
        Идет проверка на пересечение бокса попавшего в выездную зону
        в случае пересечения данных боксов
        находим дистанцию между центрами данных боксов и записываем в переменную
        обновляем флаг минимальной дистанции на True
        сохраняем текущий id машинки и ее бокс
        и в случае если текущее расстояние меньше, чем константа
        перезаписываем ее расстояние
        дальше я написал гавно, которое надо переписать, но мне лень
        если это последняя машинка
        проверяем ее на пересечение луча и линии выезда и далее
        готовим ее к удалению
        если у нас несколько машинок пересекаются боксами
        находим текущее расстояние 
        в случае если оно меньше, перезаписываем id на 
        новое совпадение, проверяем ее на пересечение луча и линии выезда и далее
        готовим к удалению
        дальше о5 написано гавно
        если у нас словарик закончился
        пихаем наше совпадение,
        проверяем ее на пересечение луча и линии выезда и далее
        готовим ее к удалению
        В случае если у нас словарик удаления не пустой, вызываем
        функцию отчищения словарика, передаем в нее id машинок под удаление
        удаляем их и возвращаем True иначе если словарик пустой просто выходим из функции
        '''
        for i, frame in self.dictCoordinate.items():            
            lastFrame = frame[-1]
            if intersection(lastFrame, deleteBoxmass[0]):                                                        
                if not minDistanceFlag:                                            
                    currentDistance = self.distanceBetweenC(lastFrame, deleteBoxmass[0])
                    if minDistance > currentDistance:
                        minDistanceFlag = True
                        currentCar = i
                        currentFrame = lastFrame
                        minDistance = currentDistance
                        if countDict == sizeDict:
                            ray.append(currentFrame)
                            ray.append(deleteBoxmass[0])
                            delDict.append(currentCar)
                else:                                                                   
                    currentDistance = self.distanceBetweenC(lastFrame, deleteBoxmass[0])
                    if minDistance > currentDistance:                         
                        minDistance = currentDistance                        
                        currentCar = i
                        currentFrame = lastFrame
                        if countDict == sizeDict:
                            ray.append(currentFrame)
                            ray.append(deleteBoxmass[0])      
                            delDict.append(currentCar)
                    else:                                                               
                        if countDict == sizeDict:
                            ray.append(currentFrame)
                            ray.append(deleteBoxmass[0])      
                            delDict.append(currentCar)
            else:        
                if minDistanceFlag and  (countDict == sizeDict):
                    ray.append(currentFrame)
                    ray.append(deleteBoxmass[0])      
                    delDict.append(currentCar)  
            countDict+=1
        if delDict:
            self.clearDictcoordinate(delDict, img)
            return True
        

  
            



    def _plot_boxes(self, output_img, labels, boxes):
        """ Функция обработки выходного изображения, получает от модельки исходное изображение,
        значение, что попало в кадр и его координаты  
        >>>straightCoef( )
        output_img
        """

        
        # Создание зоны обнаружения перекрестка
        self.coords = [(0, 960), (0, 800), (380, 350), (580, 350), (960,880), (960,960)]     #Магические числа маски 

        self.Zone1 = [(0, 960), (960, 960), (960, 750), (0, 750)]

        self.lineZone1 = [(0, 955), (5000, 955)]
        # Создание полигонов перекрестка и полигонов выездных зон
        poly = Polygon(self.coords)
        exitZone1 = Polygon(self.Zone1)
        
        # Цикл проверки всех полученных боксов

        for label, (x1, y1, x2, y2) in zip(labels, boxes):

            if (x2 - x1) * (y2 - y1) < 0.25 and label == 3 or label == 6:
                
                # Сохранение значений левого верхнего угла x1, y1 и правого нижнего x2, y2
                x1 = int(x1 * output_img.shape[1])
                y1 = int(y1 * output_img.shape[0])
                x2 = int(x2 * output_img.shape[1])
                y2 = int(y2 * output_img.shape[0])
                
                self.center_point, w, h = self.getCentroid(x1, y1, x2, y2)
                '''
                Проверка, входит ли центр блока в зону обнаружения перекрестка
                '''
                p1 = Point(self.center_point)  # Получение центра прямоугольника и запись в массив
                self.deleteBoxmass.clear()

                ''' Если центр обнаруженного бокса находится в зоне перекрестка
                и входит в полигон выезда 1, добавляем этот бокс в массив
                отрисовываем его на выходном кадре 
                если функция matchesDictcoordinate находит совпадение с машинкой в словаре
                удаляет эту машинку и следовательно возвращает True отрисовываем линию выезда
                отрисовываем прямоугольник, отрисовываем его координаты на видео. И увеличиваем
                счетчик машинок в этой зоне на 1
                Остальные if работают точно так же 
                Весь код в if можно вынести в функцию 
                '''

                # cv2.rectangle(output_img, (0, 750), (960, 960), (247, 0, 49), 2)
                # cv2.line(output_img, self.lineZone1[0], self.lineZone1[1], (247, 0, 49), 2)

                # cv2.line(output_img,(475,420) ,(270,960) , (133,213,184), 2)# Разметка
                # cv2.line(output_img,(498,420) ,(482,960) , (133,213,184), 2)# Разметка
                # cv2.line(output_img,(531,420) ,(695,960) , (133,213,184), 2)# Разметка

                if p1.within(poly):  # Если True ( Точка входит в зону обнаружения перекрестка)
                    if p1.within(exitZone1):
                        self.deleteBoxmass.append([x1, y1, x2, y2])
                        if self.matchesDictcoordinate(self.deleteBoxmass, output_img):
                            cv2.line(output_img, self.lineZone1[0], self.lineZone1[1], (0, 0, 255), 5)

                    # В случае если машинка не попала в выездные зоны, но попала на перекресток 
                    if not p1.within(exitZone1) :
                        if not len(self.dictCoordinate):  # Проверка на наличие машинок в словарике
                            self.firstFrame.append([x1, y1, x2, y2])  # Запись полученных блоков в массив первого кадра                            

                        elif len(self.dictCoordinate):
                            self.secondFrame.append([x1, y1, x2, y2])  # Иначе запись в массив второго кадра
                        cv2.rectangle(output_img, (x1,y1), (x2,y2), (0, 255, 0), 2)    
                    cv2.circle(output_img, self.center_point, 4, (0, 0, 255), 3)
                


        '''                    
        Решение проблемы когда словарь хранит больше значений чем появилось на втором кадре        
        При условии, что машинка не попала в зону выезда, а просто пропала
        В любом случае, когда мы добавляем значение в словарик, мы заполняем массив использованных машинок
        После каждой итерации, сравниваем массив второго кадра с массивом уже присвоенных машинок
        Если они совпадают, мы выходим из всех циклов
        По окончанию цикла, если массивы не совпадают
        Машинок появилось больше чем было на текущий момент в словарике
        Мы дополняем словарик
        '''
       
        self.newBoxesfdict = []
        self.usedBoxesflag = False
        self.usedBoxes.clear()
        self.unUsedboxes.clear()
        self.counterDict = len(self.dictCoordinate)
        self.counterDictin = 1
        if not self.counterDict:                                                                
            for i in range(0, len(self.firstFrame)):                                                 
                self.dictCoordinate[self.countCars] = [self.firstFrame[i]]
                self.dictCenter[self.countCars]  =  [self.getCentr(self.firstFrame[i])]  # Добавил                           
                self.countCars+=1
            
            self.firstFrame.clear()
        elif len(self.dictCoordinate):                                                              
            for i, frame in self.dictCoordinate.items():                                            
                lastFrame = frame[-1]                                                          
                self.minDistance = 90
                self.minDistanceFlag = False
                for j in range(0, len(self.secondFrame)):                                           
                    if usedBox(self.secondFrame,self.usedBoxes):
                        self.usedBoxesflag = True
                        break
                    if intersection(lastFrame, self.secondFrame[j]):                                                         
                        if not self.minDistanceFlag:                                            
                            self.minDistance = self.distanceBetweenC(lastFrame, self.secondFrame[j])
                            self.currentBox = self.secondFrame[j]
                            self.minDistanceFlag = True
                            
                            if j == len(self.secondFrame) - 1:
                                self.dictCoordinate[i].append(self.currentBox)
                                self.dictCenter[i].append(self.getCentr(self.currentBox))# Добавил
                                self.usedBoxes.append(self.currentBox)                          
                        else:                                                                   
                            self.currentDistance = self.distanceBetweenC(lastFrame, self.secondFrame[j])
                            if self.minDistance > self.currentDistance:                         
                                self.minDistance = self.currentDistance                        
                                self.currentBox = self.secondFrame[j]                                
                                if j == len(self.secondFrame) - 1:                                   
                                    self.dictCoordinate[i].append(self.currentBox)
                                    self.dictCenter[i].append(self.getCentr(self.currentBox))# Добавил                  
                                    self.usedBoxes.append(self.currentBox)
                            else:                                                               
                                if j == len(self.secondFrame) - 1:
                                    self.dictCoordinate[i].append(self.currentBox)
                                    self.dictCenter[i].append(self.getCentr(self.currentBox))# Добавил
                                    self.usedBoxes.append(self.currentBox)
                    else:                                                                       
                        if self.minDistanceFlag and j == len(self.secondFrame) - 1:
                            self.dictCoordinate[i].append(self.currentBox)
                            self.dictCenter[i].append(self.getCentr(self.currentBox))# Добавил
                            self.usedBoxes.append(self.currentBox)
                        else:
                            if self.counterDictin == self.counterDict and j == len(self.secondFrame) - 1:
                                self.usedBoxes.append(self.secondFrame[j])
                                self.newBoxesfdict.append(self.secondFrame[j])
                if self.usedBoxesflag:
                    break
                self.counterDictin+=1
                
        self.unUsedboxes = notUsedBox(self.secondFrame, self.usedBoxes)

        if not self.unUsedboxes == []:
            for boxes in self.unUsedboxes:
                self.newBoxesfdict.append(boxes)

        if self.newBoxesfdict:
            for i in self.newBoxesfdict:
                self.dictCoordinate[self.countCars] = [i]
                self.dictCenter[self.countCars] = [self.getCentr(i)]# Добавил
                self.countCars += 1

        self.secondFrame.clear()
        # self.drowTracker(output_img, self.dictCenter)
        # cv2.imshow('res', output_img)
        # cv2.waitKey(0)  
        
            
        return output_img
        

    def __call__(self, img):
        """
        Now the call method This takes a raw frame from opencv finds the boxes and draws on it.
        """
        if type(img) == np.ndarray:
            # single image case

            # First convert the image to a tensor, reverse the channels, unsqueeze and send to the right device.
            img_tens = self.tfms(Image.fromarray(img[:, :, ::-1])).unsqueeze(0).to(self.device)

            # Run the tensor through the network.
            # We'll use NVIDIAs utils to decode.
            results = utils.decode_results(self.model(img_tens))
            boxes, labels, conf = utils.pick_best(results[0], self.threshold)
            # Crop the image to match what we've been predicting on.
            output_img = self._crop_img(img)
            return self._plot_boxes(output_img, labels, boxes)

        elif type(img) == list:
            # batch case
            if len(img) == 0:
                # Catch empty batch case
                return None

            tens_batch = torch.cat([self.tfms(Image.fromarray(x[:, :, ::-1])).unsqueeze(0) for x in img]).to(
                self.device)
            results = utils.decode_results(self.model(tens_batch))

            output_imgs = []
            for im, result in zip(img, results):
                boxes, labels, conf = utils.pick_best(result, self.threshold)

                output_imgs.append(self._plot_boxes(self._crop_img(im), labels, boxes))

            return output_imgs

        else:
            raise TypeError(f"Type {type(img)} not understood")





batch_size = 64           

cap = cv2.VideoCapture('TraficPiterCut.mp4') # Исходное видео
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = min([width, height])

# SCREEN_SIZE = (640, 480)
# base = np.zeros((size, size) + (3,), dtype='uint8')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter("TraficPiterOut.mp4", fourcc, 30.0, (size, size))

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('TraficPiterOut.avi', fourcc, 30.0, (640,  480))

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (960,  960))
# Получение координат нарисованных зон и линий
obj_detect = ObjectDetectionPipeline(device="cuda", threshold=0.3)

count = 0
exit_flag = True

old_frame = None
while exit_flag:
    batch_inputs = []
    for _ in range(batch_size):
        count += 1
        print(count)
        ret, frame = cap.read()
        if ret == True:
            if old_frame is not None and (old_frame == frame).all(): # np.array_equal(old_frame, frame)
                old_frame = None
                continue
        old_frame = frame

        
        if ret:
            batch_inputs.append(frame)
        else:
            exit_flag = False
            break

    outputs = obj_detect(batch_inputs)

    
    if outputs is not None:
        for output in outputs:
            
            out.write(output)
            # cv2.imshow('frame', output)
                 
    else:
        exit_flag = False
cap.release()
out.release()