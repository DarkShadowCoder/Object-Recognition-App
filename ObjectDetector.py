import cv2
import numpy as np

class ObjectDetection():

    
    def __init__(self,video = 0 , img = None):
        self.video = video
        self.img = img
        self.net = cv2.dnn.readNet("data/model/yolov4.weights" , "data/model/yolov4.cfg")
        self.classes = ["person" , 'bicycle','car','motorbike','aeroplane','bus','train','truck','boat','traffic light' , 'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','sofa','pottedplant','bed','dinningtable','toilet','tvmonitor','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigirator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']

        
    def detector(self):
        if self.video is None:
            return self.detectorImg(self.img)
        elif isinstance(self.video , str):
            return self.detectorVideo(self.video)
        else:
            return self.detectorVideo(path=0)
    
    def detectorVideo(self , path):
        cap = cv2.VideoCapture(path)
        #img = cv2.imread('data/images/detection.jpg')
        while True:
            _, img = cap.read()
            height , width,_ = img.shape

            blob = cv2.dnn.blobFromImage(img , 1/255 , (416,416),(0,0,0), swapRB = True , crop = False)
            self.net.setInput(blob)
            output_layers_names = self.net.getUnconnectedOutLayersNames()
            layersOutputs = self.net.forward(output_layers_names)

            boxes = []
            confidences = []
            class_ids = []

            for output in layersOutputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0]*width)
                        center_y = int(detection[1]*height)
                        w = int(detection[2]*width)
                        h = int(detection[3]*height)

                        x = int(center_x - w/2)
                        y = int(center_y - h/2)

                        boxes.append([x,y,w,h])
                        confidences.append((float(confidence)))
                        class_ids.append(class_id)


            indexes = cv2.dnn.NMSBoxes(boxes , confidences, 0.5 , 0.4)
            font = cv2.FONT_HERSHEY_PLAIN
            colors = np.random.uniform(0,255,size=(len(boxes) , 3))
            print("indexes:" , indexes)
            if len(indexes) > 0:
                for i in indexes.flatten():
                    x,y,w,h = boxes[i]
                    label = str(self.classes[class_ids[i]])
                    confidence = str(round(confidences[i], 2))
                    color = colors[i]
                    cv2.rectangle(img , (x,y) , (x+w ,  y+h)  , color , 2 )
                    cv2.putText(img , label + " " + confidence , (x,y + 5) , font , 1 , color , 2)
            
            cv2.imshow('Image' , img)
            key = cv2.waitKey(1)
            if key == ord('J'):
                break
        


    def detectorImg(self , image):
        img = cv2.imread(image)
        height , width,_ = img.shape
        
        blob = cv2.dnn.blobFromImage(img , 1/255 , (416,416),(0,0,0), swapRB = True , crop = False)
        self.net.setInput(blob)
        output_layers_names = self.net.getUnconnectedOutLayersNames()
        layersOutputs = self.net.forward(output_layers_names)

        boxes = []
        confidences = []
        class_ids = []

        for output in layersOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)

                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    boxes.append([x,y,w,h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes , confidences, 0.5 , 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0,255,size=(len(boxes) , 3))
        print("indexes:" , indexes)
        if len(indexes) > 0:
            for i in indexes.flatten():
                x,y,w,h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                color = colors[i]
                cv2.rectangle(img , (x,y) , (x+w ,  y+h)  , color , 2 )
                cv2.putText(img , label + " " + confidence , (x,y + 5) , font , 1 , color , 2)
        cv2.imshow('Image' , img)
