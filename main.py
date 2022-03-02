import speech_recognition as sr
import pyttsx3
import cv2
import threading
import datetime
import wikipedia
import pyjokes
import mediapipe as mp
import numpy as np
import os
import webbrowser
from tkinter import *
from TKlighter import *
from tkinter import filedialog
from matplotlib import pyplot

from mediapipe.python.solutions.hands_connections import HAND_CONNECTIONS

mp_face = mp.solutions.face_detection
mpFace = mp_face.FaceDetection()

mp_hand = mp.solutions.hands
mpHand = mp_hand.Hands()

mpDraw = mp.solutions.drawing_utils

wikiSearch = ["who is", "where is", "what is", "how to"]

class EclipsaIDE:
    def __init__(self):
        def syntaxHightLight(evnt):
            def_h(editor, "lime")
            function_h(editor, "red")
            True_h(editor, "red")
            False_h(editor, "red")
            import_h(editor, "red")
            from_h(editor, "red")
            if_h(editor, "red")
            else_h(editor, "red")
            elif_h(editor, "red")
            try_h(editor, "red")
            except_h(editor, "red")
            for_h(editor, "red")
            while_h(editor, "red")
            yield_h(editor, "red")
            pass_h(editor, "red")
            break_h(editor, "red")
            continue_h(editor, "red")
            with_h(editor, "red")
            del_h(editor, "red")
            open_h(editor, "red")
            as_h(editor, "red")
            in_h(editor, "red")
            or_h(editor, "red")
            and_h(editor, "red")
            class_h(editor, "lime")
            print_h(editor, "red")
            int_h(editor, "red")
            float_h(editor, "red")
            str_h(editor, "red")
            return_h(editor, "red")
            input_h(editor, "red")
            None_h(editor, "red")
            global_h(editor, "red")
            Exception_h(editor, "red")

        def run():
            exec(editor.get("1.0", END))
        
        filename = ""
        _filetypes = [
            ('Text', '*.txt'),
                ('All files', '*'),
                ]

        def openFile():
            filename = filedialog.askopenfilename(initialdir = "/",
                                                title = "Select a File",
                                                filetypes = (("Text files",
                                                                "*.txt*"),
                                                            ("all files",
                                                                "*.*")))
        
        def save_file_as():
            filename = filedialog.asksaveasfilename(filetypes=_filetypes)
            f = open(filename, 'w')
            f.write(editor.get('1.0', 'end'))
            f.close()
        
        def save_file():
            if (filename == ""):
                save_file_as()
            else:
                f = open(filename, 'w')
                f.write(editor.get('1.0', 'end'))
                f.close()

        top = Tk()
        top.title("Eclipsa-IDE")
        top.configure(bg="#515151")

        top.bind('<Control-s>', save_file)

        _font = ("Consolas", 12, "bold")

        editor = Text(insertbackground="red", bg="#505050", fg="white")
        editor.configure(font=_font)
        editor.bind("<Key>", syntaxHightLight)

        menuBar = Menu(top)

        FILE = Menu(menuBar)
        FILE.add_command(label="Open", command=openFile)

        RUN = Menu(menuBar, tearoff=0)
        RUN.add_command(label="Run", command=run)

        menuBar.add_cascade(label="File", menu=FILE)
        menuBar.add_cascade(label="Run", menu=RUN)

        editor.pack(fill=BOTH, expand=True)

        top.config(menu=menuBar)
        top.mainloop()

def talk(cmd):
    pytt = pyttsx3.init()
    # voices = pytt.getProperty('voices')
    # pytt.setProperty('voice', voices[1].id)
    pytt.setProperty("rate", 165)
    pytt.say(cmd)
    print(cmd)
    pytt.runAndWait()

def op_webcam():
    capture = cv2.VideoCapture(0)
    while True:
        success, image = capture.read()
        cv2.imshow("Webcam", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            capture.release()
            cv2.destroyAllWindows()

def faceDetetion():
    # https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
    FACE_PROTO = "weights/deploy.prototxt.txt"
    # https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
    FACE_MODEL = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    # The gender model architecture
    # https://drive.google.com/open?id=1W_moLzMlGiELyPxWiYQJ9KFaXroQ_NFQ
    GENDER_MODEL = 'weights/deploy_gender.prototxt'
    # The gender model pre-trained weights
    # https://drive.google.com/open?id=1AW3WduLk1haTVAxHOkVS_BEzel1WXQHP
    GENDER_PROTO = 'weights/gender_net.caffemodel'
    # Each Caffe Model impose the shape of the input image also image preprocessing is required like mean
    # substraction to eliminate the effect of illunination changes
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    # Represent the gender classes
    GENDER_LIST = ['Male', 'Female']
    # The model architecture
    # download from: https://drive.google.com/open?id=1kiusFljZc9QfcIYdU2s7xrtWHTraHwmW
    AGE_MODEL = 'weights/deploy_age.prototxt'
    # The model pre-trained weights
    # download from: https://drive.google.com/open?id=1kWv0AjxGSN0g31OeJa02eBGM0R_jcjIl
    AGE_PROTO = 'weights/age_net.caffemodel'
    # Represent the 8 age classes of this CNN probability layer
    AGE_INTERVALS = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)',
                    '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
    # Initialize frame size
    frame_width = 1280
    frame_height = 720
    # load face Caffe model
    face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
    # Load age prediction model
    age_net = cv2.dnn.readNetFromCaffe(AGE_MODEL, AGE_PROTO)
    # Load gender prediction model
    gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)

    def get_faces(frame, confidence_threshold=0.5):
        # convert the frame into a blob to be ready for NN input
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
        # set the image as input to the NN
        face_net.setInput(blob)
        # perform inference and get predictions
        output = np.squeeze(face_net.forward())
        # initialize the result list
        faces = []
        # Loop over the faces detected
        for i in range(output.shape[0]):
            confidence = output[i, 2]
            if confidence > confidence_threshold:
                box = output[i, 3:7] * \
                    np.array([frame.shape[1], frame.shape[0],
                            frame.shape[1], frame.shape[0]])
                # convert to integers
                start_x, start_y, end_x, end_y = box.astype(np.int)
                # widen the box a little
                start_x, start_y, end_x, end_y = start_x - \
                    10, start_y - 10, end_x + 10, end_y + 10
                start_x = 0 if start_x < 0 else start_x
                start_y = 0 if start_y < 0 else start_y
                end_x = 0 if end_x < 0 else end_x
                end_y = 0 if end_y < 0 else end_y
                # append to our list
                faces.append((start_x, start_y, end_x, end_y))
        return faces

    # from: https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
    def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]
        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image
        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)
        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))
        # resize the image
        return cv2.resize(image, dim, interpolation = inter)


    def get_gender_predictions(face_img):
        blob = cv2.dnn.blobFromImage(
            image=face_img, scalefactor=1.0, size=(227, 227),
            mean=MODEL_MEAN_VALUES, swapRB=False, crop=False
        )
        gender_net.setInput(blob)
        return gender_net.forward()


    def get_age_predictions(face_img):
        blob = cv2.dnn.blobFromImage(
            image=face_img, scalefactor=1.0, size=(227, 227),
            mean=MODEL_MEAN_VALUES, swapRB=False
        )
        age_net.setInput(blob)
        return age_net.forward()



    def predict_age_and_gender():
        """Predict the gender of the faces showing in the image"""
        # Initialize frame size
        # frame_width = 1280
        # frame_height = 720
        # Read Input Image
        c = cv2.VideoCapture(0)
        while True:
            success, img =c.read()
            # resize the image, uncomment if you want to resize the image
            # img = cv2.resize(img, (frame_width, frame_height))
            # Take a copy of the initial image and resize it
            frame = img.copy()
            if frame.shape[1] > frame_width:
                frame = image_resize(frame, width=frame_width)
            # predict the faces
            faces = get_faces(frame)
            # Loop over the faces detected
            # for idx, face in enumerate(faces):
            for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
                face_img = frame[start_y: end_y, start_x: end_x]
                age_preds = get_age_predictions(face_img)
                gender_preds = get_gender_predictions(face_img)
                i = gender_preds[0].argmax()
                gender = GENDER_LIST[i]
                gender_confidence_score = gender_preds[0][i]
                i = age_preds[0].argmax()
                age = AGE_INTERVALS[i]
                age_confidence_score = age_preds[0][i]
                # Draw the box
                label = f"{gender}-{gender_confidence_score*100:.1f}%, {age}-{age_confidence_score*100:.1f}%"
                # label = "{}-{:.2f}%".format(gender, gender_confidence_score*100)
                print(label)
                yPos = start_y - 15
                while yPos < 15:
                    yPos += 15
                box_color = (255, 0, 0) if gender == "Male" else (147, 20, 255)
                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 2)
                # Label processed image
                font_scale = 0.54
                cv2.putText(frame, label, (start_x, yPos),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_color, 2)
            cv2.imshow("Eclipsa-Vesion-Face-Detection")
            cv2.waitKey(1)

                # Display processed image
            # Cleanup
            cv2.destroyAllWindows()

            predict_age_and_gender()

def handDetection():
    c = cv2.VideoCapture(0)
    while True:
        success, image = c.read()
        cvtImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = mpHand.process(cvtImage)
        if result.multi_hand_landmarks:
            for i in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(image, i, HAND_CONNECTIONS)
        cv2.imshow("Eclipsa-Vesion-Hand-Detection", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            c.release()
            cv2.destroyAllWindows()

def eclipsa_hey_bro():
    listener = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("...")
            voice = listener.listen(source)
            command = listener.recognize_google(voice)
            print(command, "\U0001F600")
            if "hey bro" in command:
                cmd = command.replace("hey bro", "")
                if "call" in cmd:
                    if "my mum" not in cmd or "my dad" in cmd:
                        cmd = cmd.replace("call my", "calling your")
                        talk(cmd)
                    else:
                        cmd = cmd.replace("call my", "calling your")
                        talk(cmd)
                if "turn the lights off" in command:
                    cmd = "turning off"
                    talk(cmd)
                if "webcam" in command:
                    cmd = "turning on webcam"
                    talk(cmd)
                    t = threading.Thread(target=op_webcam)
                    t.start()
                    t.join()
                if "open my area" in cmd:
                    cmd = cmd.replace("open my area", "Opening python area")
                    talk(cmd)
                    EclipsaIDE()
                if "what's the date now" in cmd:
                    time = datetime.datetime.now().strftime("%H:%M")
                    talk("Cur time is", time)
                if "programming joke" in cmd:
                    talk(pyjokes.get_joke())
                if "detect my face" in cmd:
                    talk("Ok, i'll try")
                    faceDetetion()
                if "try to detect my hands" in cmd:
                    talk("Ok, i'll try")
                    handDetection()
                if "search for" in cmd:
                    cmd = cmd.replace("search for", "")
                    talk(f"searching for {cmd}")
                    webbrowser.open(f"http://{cmd}")
                elif "where is" in cmd:
                    cmd = cmd.replace("where is", "")
                    info = wikipedia.summary(cmd, 2)
                    talk(info)
                elif "who is" in cmd:
                    cmd = cmd.replace("who is", "")
                    info = wikipedia.summary(cmd, 1)
                    talk(info)
                elif "what is" in cmd:
                    cmd = cmd.replace("what is", "")
                    info = wikipedia.summary(cmd, 2)
                    talk(info)
                elif "shut down" in cmd:
                    os.system("shutdown /s /t 30")
    except:
        pass

if __name__ == "__main__":
    while True:
        eclipsa_hey_bro()
