import kivy
import cv2
import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.filechooser import FileChooser
from kivy.uix.video import Video
from kivy.uix.screenmanager import ScreenManager , Screen
from kivy.lang import Builder


class RoundButton(Button):
    pass

class FirstPage(Screen):
    def __init__(self, **kwargs):
        super(FirstPage ,self).__init__(**kwargs)
        # Utilisation d'un boutton personnalisé
        
        layout = BoxLayout(orientation = 'vertical')
        background = Image(source = "data/images/wallpaper.jpeg")
        layout.add_widget(background)
        button = Button(text = 'Démarrer' , background_color = (0,0,0.5,0.5) , size_hint =(None , None) , size = (150,100) , font_size = "18px" , color = "ffffffff")
        button.pos_hint = {'center_x':0.5,'center_y':0.5}
        button.bind(on_release = self.switch_to_second_page)
        layout.add_widget(button)
        self.add_widget(layout)
    
    def switch_to_second_page(self, *args):
        self.manager.current = 'second'
    
class MainApp(App):
    def build(self):
        screen_manager = ScreenManager()
        screen_manager.add_widget(FirstPage(name = 'first'))
        screen_manager.add_widget(SecondPage(name = 'second'))
        return screen_manager

# Interface Graphique principal
class SecondPage(Screen):
    def __init__(self , **kwargs):
        super(SecondPage  , self).__init__(**kwargs)
        #nav = BoxLayout(orientation = 'horizontal')
        #nav.l
        layout = BoxLayout(orientation='vertical')
        self.nav = Button(background_color = (0,0,0.9,1) , font_size = "16px" , font_family = 'Cambria' , size_hint = (1 , 0.1))
        layout.add_widget(self.nav)
        # Créer une zone de capture vidéo
        self.camera = cv2.VideoCapture(0)
        self.image = Image()
        self.image.size_hint = (1,0.75)
        # Lancer la mise à jour de l'image du flux de video
        Clock.schedule_interval(self.update , 1.0/30.0)
        layout.add_widget(self.image)

        # Créer un conteneur pour les boutons
        button_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))

        # Bouton pour enregistrer la vidéo
        record_button = Button(text='Capturer')
        record_button.font_family = "Cambria"
        record_button.bind(on_release=self.record_image)
        record_button.background_color = (0,0,0.5,0.5)
        button_layout.add_widget(record_button)

        # Bouton pour importer une photo
        import_photo_button = Button(text='Importer une photo')
        import_photo_button.bind(on_release=self.import_photo)
        import_photo_button.background_color = (0,0,0.5,0.5)
        record_button.font_family = "Cambria"
        button_layout.add_widget(import_photo_button)

        # Bouton pour importer une vidéo
        import_video_button = Button(text='Importer une vidéo')
        import_video_button.bind(on_release=self.import_video)
        import_video_button.background_color = (0,0,0.5,0.5)
        record_button.font_family = "Cambria"
        button_layout.add_widget(import_video_button)

        # Bouton pour la detection et la Reconnaissance d'objets
        self.verify_button = Button(text="Analyse en cours" , font_size = "18px" )
        self.verify_button.color = (0,0,0,1)
        self.verify_button.on_touch_down = self.change_color
        self.verify_button.on_release = self.verify
        self.verify_button.background_color = (0,1,0,1)
        record_button.font_family = "Cambria"
        button_layout.add_widget(self.verify_button)

        layout.add_widget(button_layout)
        self.add_widget(layout)
    
    def update(self,dt):
        
        net = cv2.dnn.readNet("data/model/yolov4.weights" , "data/model/yolov4.cfg")
        classes = ["pearson" , 'bicycle','car','motorbike','aeroplane','bus','train','truck','boat','traffic light' , 'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','sofa','pottedplant','bed','dinningtable','toilet','tvmonitor','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigirator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']

        ret ,frame = self.camera.read()
        height , width , _ = frame.shape
        
        if ret:

            # Initialisation des variables pour le systeme
            blob = cv2.dnn.blobFromImage(frame , 1/255 , (416,416),(0,0,0), swapRB = True , crop = False)
            net.setInput(blob)
            output_layers_names = net.getUnconnectedOutLayersNames()
            layersOutputs = net.forward(output_layers_names)
            
            boxes = []
            confidences = []
            class_ids = []

            # Localisation des objets et creation du rectangle de reconnaissance
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

            # Detection et reconnaissance des objets sur l'image
            self.indexes = cv2.dnn.NMSBoxes(boxes , confidences, 0.5 , 0.4)
            font = cv2.FONT_HERSHEY_PLAIN
            colors = np.random.uniform(0,255,size=(len(boxes) , 3))
            self.nav.text = "Nombre d'objets reconnus : " + str(len(self.indexes)) + " objets"
            print(len(self.indexes))
            if len(self.indexes) > 0:
                for i in self.indexes.flatten():
                    x,y,w,h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = str(round(confidences[i], 3)*100)
                    color = colors[i]
                    cv2.rectangle(frame , (x,y) , (x+w ,  y+h)  , color , 2 )
                    cv2.putText(frame , label + ": " + confidence + "%" , (x+10,y-10) , font , 1 , color , 2)

            # Convertir l'image en format kivy et l'afficher
            texture = self.get_texture(frame)
            self.image.texture = texture

    
    def get_texture(self , frame):
        # Convertir l'image OpenCV en image Kivy
        frame = cv2.flip(frame , 0)
        buffer = frame.tostring()
        texture = Texture.create(size = (frame.shape[1] , frame.shape[0]),colorfmt = 'bgr')
        texture.blit_buffer(buffer , colorfmt = 'bgr' , bufferfmt = 'ubyte')
        return texture
    
    def on_stop(self):
        self.camera.release()


    def verify(self):
        self.verify_button.background_color = (0.9,0,0,0.8)
        #Video = ObjectDetection.detector()
        self.camera.clear_widgets()
        self.camera = Video

    def change_color(self , widget:Button):
        widget.background_color = (0.8,0,0,0.7)
    def record_image(self, *args):
        self.camera.export_to_png("data/images/screenshot.png")
        print("Image Capturée")

    def import_photo(self, instance):
        file_chooser = FileChooser()
        file_chooser.bind(on_submit=self.display_image)
        file_chooser.open()

    def import_video(self, instance):
        file_chooser = FileChooser()
        file_chooser.bind(on_submit=self.display_video)
        file_chooser.open()

    def display_image(self, instance):
        if instance.selection and instance.selection[0].endswith(('.png', '.jpg', '.jpeg')):
            image = Image(source=instance.selection[0])
            self.camera.clear_widgets()
            self.camera.add_widget(image)

    def display_video(self, instance):
        if instance.selection and instance.selection[0].endswith(('.mp4', '.avi', '.mkv')):
            video = Video(source=instance.selection[0], play=True, options={'allow_stretch': True})
            self.camera.clear_widgets()
            self.camera.add_widget(video)



if __name__ == '__main__':
    MainApp().run()
