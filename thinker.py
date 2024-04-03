from pathlib import Path
from collections import Counter
from PIL import Image, ImageDraw
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QLabel, QFileDialog
from PyQt5 import uic
from PyQt5.QtGui import QPixmap

import pickle
import face_recognition
import sys
import os
import io


class UI(QMainWindow):

    DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
    RECOGNIZEPATH = Path("unknown")
    BOUNDING_BOX_COLOR = "blue"
    TEXT_COLOR = "white"
    IMAGEQUEUE = []
    IMAGESPOT = 0
    
    def __init__(self):
        super(UI, self).__init__()

        # Load the ui file
        uic.loadUi("gui\\notAiThinkerGUI.ui", self)

        
        # Define Widgets
        self.buttonValidate = self.findChild(QPushButton, "pushButton")
        self.buttonImages = self.findChild(QPushButton, "pushButton_2")
        self.buttonRecognize = self.findChild(QPushButton, "pushButton_3")
        self.buttonNext = self.findChild(QPushButton, "pushButton_4")
        self.buttonEncode = self.findChild(QPushButton, "pushButton_5")
        self.buttonBack = self.findChild(QPushButton, "pushButton_6")
        self.label = self.findChild(QLabel, "label")
        self.posLabel = self.findChild(QLabel, "label_2")

        # Set Buttons
        self.buttonImages.clicked.connect(self.changeImages)
        self.buttonEncode.clicked.connect(self.encode_known_faces)
        self.buttonValidate.clicked.connect(self.validate)
        self.buttonRecognize.clicked.connect(self.wrapper)
        self.buttonNext.clicked.connect(self.clickerNext)
        self.buttonBack.clicked.connect(self.clickerBack)
                        
        # Show The App
        self.show()

        # Set placeholder
        self.setDefault()


    def changeImages(self):
        fname = QFileDialog.getOpenFileName(self, "Open File", "training", "All Files (*);;PNG Files (*.png);;Jpg Files (*.jpg)")

    def setDefault(self):
        self.IMAGESPOT = 0
        self.IMAGEQUEUE.clear()
        self.processImage(Image.open("main.PNG"))

        self.pixmap = QPixmap()
        self.pixmap.loadFromData(self.IMAGEQUEUE[0])
        self.label.setPixmap(self.pixmap)


    def clickerNext(self):
        if(len(self.IMAGEQUEUE)>1 and self.IMAGESPOT < len(self.IMAGEQUEUE)-1):
            self.IMAGESPOT += 1
            self.resetLabel()
            self.pixmap.loadFromData(self.IMAGEQUEUE[self.IMAGESPOT])
            self.label.setPixmap(self.pixmap)

    def clickerBack(self):
        if(len(self.IMAGEQUEUE)>1 and self.IMAGESPOT > 0):
            self.IMAGESPOT -= 1
            self.resetLabel()
            self.pixmap.loadFromData(self.IMAGEQUEUE[self.IMAGESPOT])
            self.label.setPixmap(self.pixmap)


    def wrapper(self):

        self.setDefault()

        dirlist = os.listdir(self.RECOGNIZEPATH)

        for x in dirlist:
            self.recognize_faces("unknown/" + x)

        self.resetLabel()


    def resetLabel(self):

        self.posLabel.setText(str(self.IMAGESPOT+1) + " / " + str(len(self.IMAGEQUEUE)))

          
    def encode_known_faces(self, model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH):
        names = []
        encodings = []
        for filepath in Path("training").glob("*/*"):
            name = filepath.parent.name
            image = face_recognition.load_image_file(filepath)

            face_locations = face_recognition.face_locations(image, model=model)
            face_encodings = face_recognition.face_encodings(image, face_locations)

            for encoding in face_encodings:
                names.append(name)
                encodings.append(encoding)

        name_encodings = {"names": names, "encodings": encodings}
        with encodings_location.open(mode="wb") as f:
            pickle.dump(name_encodings, f)


    def recognize_faces(self, image_location: str, model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH):
        
        with encodings_location.open(mode="rb") as f:
            loaded_encodings = pickle.load(f)

        input_image = face_recognition.load_image_file(image_location)

        input_face_locations = face_recognition.face_locations(input_image, model=model)

        input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)

        pillow_image = Image.fromarray(input_image)

        draw = ImageDraw.Draw(pillow_image)

        for bounding_box, unknown_encoding in zip(input_face_locations, input_face_encodings):

            name = self._recognize_face(unknown_encoding, loaded_encodings)

            if not name:
                name = "Unknown"
            self._display_face(draw, bounding_box, name)
        
        del draw

        buf = io.BytesIO()
        pillow_image.save(buf, format='jpeg')

        pillow_image2 = Image.open(buf)

        self.processImage(pillow_image2)

    
    def _recognize_face(self, unknown_encoding, loaded_encodings):

        boolean_matches = face_recognition.compare_faces(loaded_encodings["encodings"], unknown_encoding)
        
        votes = Counter(name for match, name in zip(boolean_matches, loaded_encodings["names"]) if match)

        if votes:

            return votes.most_common(1)[0][0]
        

    def _display_face(self, draw, bounding_box, name):
        top, right, bottom, left = bounding_box

        draw.rectangle(((left, top), (right, bottom)), outline=self.BOUNDING_BOX_COLOR)

        text_left, text_top, text_right, text_bottom = draw.textbbox((left, bottom), name)

        draw.rectangle(((text_left, text_top), (text_right, text_bottom)), fill="blue", outline="blue")

        draw.text((text_left, text_top), name, fill="white")


    def image_to_byte_array(self, image:Image):
        imgByteArr = io.BytesIO()
        image.save(imgByteArr, format=image.format)
        imgByteArr = imgByteArr.getvalue()
        return imgByteArr


    def validate(self, model: str = "hog"):

        self.setDefault()

        for filepath in Path("validation").rglob("*"):

            if filepath.is_file():

                self.recognize_faces(image_location=str(filepath.absolute()), model=model)
        
        self.resetLabel()


    def processImage(self, image:Image):
        
        image.thumbnail((1100,1100), Image.LANCZOS)
        image2 = self.image_to_byte_array(image)
        self.IMAGEQUEUE.append(image2)

        
app = QApplication(sys.argv)
UIWindow = UI()
app.exec_()