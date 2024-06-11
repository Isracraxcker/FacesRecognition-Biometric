# librerias
from tkinter import *
import cv2
import face_recognition as fr
import numpy as np
from PIL import Image, ImageTk
import imutils
import math
import mediapipe as mp
import os


# Face Code
def Code_Face(images):
    listacod = []

    # Iteramos
    for img in images:
        # Correccion de color
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Codificamos la imagen
        cod = fr.face_encodings(img)[0]
        # Almacenamos
        listacod.append(cod)

    return listacod

# Close function
def Close_Window():
    global step, conteo
    #Reset
    conteo = 0
    step = 0
    pantalla2.destroy()

# Close2
def Close_Window2():
    global step, conteo
    # Reset Variables
    conteo = 0
    step = 0
    pantalla3.destroy()



def Profile():
    global step, conteo, UserName, OutFolderPathUser
    # Reset Variables
    conteo = 0
    step = 0

    pantalla4 = Toplevel(pantalla)
    pantalla4.title("PROFILE")
    pantalla4.geometry("1280x720")



    back = Label(pantalla4, image=imagenB, text="Back")
    back.place(x=0, y=0, relwidth=1, relheight=1)

    # Archivo
    UserFile = open(f"{OutFolderPathUser}/{UserName}.txt", 'r')
    InfoUser = UserFile.read().split(',')
    Name = InfoUser[0]
    User = InfoUser[1]
    Pass = InfoUser[2]
    UserFile.close()

    # Check
    if User in clases:
        # Interfaz
        texto1 = Label(pantalla4, text=f"BIENVENIDO {Name}", font=("Helvetica", 24, "bold"), fg="white", bg="#2C2F33")
        texto1.place(relx=0.5, y=50, anchor="center")
        # Label
        # Video
        lblImgUser = Label(pantalla4, bg="#2C2F33")
        lblImgUser.place(relx=0.5, rely=0.5, anchor="center")

        # Imagen
        PosUserImg = clases.index(User)
        UserImg = images[PosUserImg]
        ImgUser = Image.fromarray(UserImg)

        # Leer y procesar la imagen del usuario
        ImgUser = cv2.imread(f"{OutFolderPathFace}/{User}.png")
        ImgUser = cv2.cvtColor(ImgUser, cv2.COLOR_RGB2BGR)
        ImgUser = Image.fromarray(ImgUser)

        # Configurar la imagen en el label
        IMG = ImageTk.PhotoImage(image=ImgUser)

        lblImgUser.configure(image=IMG)
        lblImgUser.image = IMG


# Sign Biometric Function
def Sign_Biometric():
    global LogUser, LogPass, OutFolderPathFace, cap, lblVideo, pantalla3, FaceCode, clases, images, pantalla2, step, parpadeo, conteo, UserName

# check Cap
    if cap is not None:
        ret, frame = cap.read()

        frameSave = frame.copy()
        # Resize
        frame = imutils.resize(frame, width=1280)

        # RGB
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        # Frame Show
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if ret == True:
            # Inference Face Mesh
            res = FaceMesh.process(frameRGB)


            #Result List
            px = []
            py = []
            lista = []
            if res.multi_face_landmarks:
                #Extract Face Mesh
                for rostros in res.multi_face_landmarks:
                    mpDraw.draw_landmarks(frame, rostros, FacemeshObject.FACEMESH_CONTOURS, ConfigDraw, ConfigDraw)

                    #Extract KeyPoint
                    for id, puntos in enumerate(rostros.landmark):

                        # Info Image
                        al, an, c = frame.shape
                        x, y = int(puntos.x * an), int(puntos.y * al)
                        px.append(x)
                        py.append(y)
                        lista.append([id, x, y])

                        # 468 KeyPoints
                        if len(lista) == 468:
                            #Ojo Derecho
                            x1,y1 = lista[145][1:]
                            x2, y2 = lista[159][1:]
                            longitud1 = math.hypot(x2-x1, y2-y1)

                            # Ojo Izquierdo
                            x3, y3 = lista[374][1:]
                            x4, y4 = lista[386][1:]
                            longitud2 = math.hypot(x4 - x3, y4 - y3)

                            # Parietal Derecho
                            x5, y5 = lista[139][1:]
                            # Parietal Izquierdo
                            x6, y6 = lista[368][1:]

                            # Ceja Derecha
                            x7, y7 = lista[70][1:]
                            x8, y8 = lista[300][1:]

                            # Face Detect
                            faces = detector.process(frameRGB)

                            if faces.detections is not None:
                                for face in faces.detections:

                                    # Bbox: " ID, BBOX, SCORE"
                                    score = face.score
                                    score = score[0]
                                    bbox = face.location_data.relative_bounding_box

                                    # Threshold
                                    if score > confThreshold:
                                        # Pixeles
                                        xi, yi, anc, alt = bbox.xmin, bbox.ymin, bbox.width, bbox.height
                                        xi, yi, anc, alt = int( xi * an), int(yi * al), int(anc * an), int(alt * al)

                                        # Offset X
                                        offsetan = (offsetx / 100 ) * anc
                                        xi = int(xi - int(offsetan / 2))
                                        anc = int(anc + offsetan)
                                        xf = xi + anc

                                        # Offset Y
                                        offsetal = (offsety / 100) * alt
                                        yi = int(yi - offsetal)
                                        alt = int(alt + offsetal)
                                        yf = yi + alt

                                        # ERROR

                                        if xi < 0: xi = 0
                                        if y1 < 0: yi = 0
                                        if anc < 0: anc = 0
                                        if alt < 0: alt = 0

                                        # Steps

                                        if step == 0:
                                            # Draw
                                            cv2.rectangle(frame, (xi, yi, anc, alt), (255, 0, 255), 2)

                                            #IMG Step 0
                                            als0, ans0, c = img_step0.shape
                                            frame[50:50 +als0, 50:50 + ans0] = img_step0

                                            # IMG Step 1
                                            als1, ans1, c = img_step1.shape
                                            frame[50:50 + als1, 1030:1030 + ans1] = img_step1

                                            # IMG Step 2
                                            als2, ans2, c = img_step2.shape
                                            frame[270:270 + als2, 1030:1030 + ans2] = img_step2

                                            # Face Center
                                            if x7 > x5 and x8 < x6:
                                                # IMG Check
                                                alch, anch, c = img_check.shape
                                                frame[165:165 + alch, 1105:1105 + anch] = img_check

                                                # Conteo Parpadeo
                                                if longitud1 <= 10 and longitud2 <= 10 and parpadeo == False:
                                                    conteo = conteo + 1
                                                    parpadeo = True

                                                elif longitud2 > 10 and longitud2 > 10 and parpadeo == True:
                                                    parpadeo = False
                                                    # Parpadeos
                                                    # Conteo de parpadeos
                                                cv2.putText(frame, f'Parpadeos: {int(conteo)}', (1070, 375),cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                                                (255, 255, 255), 1)

                                                #Conteo

                                                if conteo >= 3:
                                                    # IMG Check
                                                    alch, anch, c = img_check.shape
                                                    frame[385:385 + alch, 1105:1105 + anch] = img_check

                                                    # Open Eyes
                                                    if longitud1 > 14 and longitud2 > 14:
                                                        # Cerramos
                                                        step = 1

                                            else:
                                                conteo = 0

                                        if step == 1:
                                            # Draw
                                            cv2.rectangle(frame, (xi, yi, an, al), (0, 255, 0), 2)
                                            # IMG check Liveness
                                            allich, anlich, c = img_livCheck.shape
                                            frame[50:50 + allich, 50:50 + anlich] = img_livCheck

                                            # Find faces
                                            facess = fr.face_locations(frameRGB)
                                            facescod = fr.face_encodings(frameRGB, facess)

                                            # Iteramos
                                            for facescod, facesloc in zip(facescod, facess):

                                                # Matcching
                                                Match = fr.compare_faces(FaceCode, facescod)

                                                # Similitudes
                                                simi = fr.face_distance(FaceCode, facescod)

                                                # Min
                                                min = np.argmin(simi)

                                                if Match[min]:
                                                    # UserName
                                                    UserName = clases[min].upper()

                                                    Profile()

                            # Close
                            close = pantalla3.protocol("WM_DELETE_WINDOW", Close_Window2)



                            #Circles
                            #cv2.circle(frame, (x7,y7), 2, (255,0,0), cv2.FILLED)
                            #cv2.circle(frame, (x8, y8), 2, (255, 0, 0), cv2.FILLED)

            # Rendimensionamos el video
            frame = imutils.resize(frame, width=1280)

            # Convertimos el video
            im = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=im)

            # Mostramos en el GUI
            lblVideo.configure(image=img)
            lblVideo.image = img
            lblVideo.after(10, Sign_Biometric)

        else:
            cap.release()

# Log Biometric
def Log_Biometric():
    global pantalla2, conteo, parpadeo, img_info, step, cap, lblVideo, RegUser

    # check Cap
    if cap is not None:
        ret, frame = cap.read()

        frameSave = frame.copy()
        # Resize
        frame = imutils.resize(frame, width=1280)

        # RGB
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        # Frame Show
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if ret == True:
            # Inference Face Mesh
            res = FaceMesh.process(frameRGB)


            #Result List
            px = []
            py = []
            lista = []
            if res.multi_face_landmarks:
                #Extract Face Mesh
                for rostros in res.multi_face_landmarks:
                    mpDraw.draw_landmarks(frame, rostros, FacemeshObject.FACEMESH_CONTOURS, ConfigDraw, ConfigDraw)

                    #Extract KeyPoint
                    for id, puntos in enumerate(rostros.landmark):

                        # Info Image
                        al, an, c = frame.shape
                        x, y = int(puntos.x * an), int(puntos.y * al)
                        px.append(x)
                        py.append(y)
                        lista.append([id, x, y])

                        # 468 KeyPoints
                        if len(lista) == 468:
                            #Ojo Derecho
                            x1,y1 = lista[145][1:]
                            x2, y2 = lista[159][1:]
                            longitud1 = math.hypot(x2-x1, y2-y1)

                            # Ojo Izquierdo
                            x3, y3 = lista[374][1:]
                            x4, y4 = lista[386][1:]
                            longitud2 = math.hypot(x4 - x3, y4 - y3)

                            # Parietal Derecho
                            x5, y5 = lista[139][1:]
                            # Parietal Izquierdo
                            x6, y6 = lista[368][1:]

                            # Ceja Derecha
                            x7, y7 = lista[70][1:]
                            x8, y8 = lista[300][1:]

                            # Face Detect
                            faces = detector.process(frameRGB)

                            if faces.detections is not None:
                                for face in faces.detections:

                                    # Bbox: " ID, BBOX, SCORE"
                                    score = face.score
                                    score = score[0]
                                    bbox = face.location_data.relative_bounding_box

                                    # Threshold
                                    if score > confThreshold:
                                        # Pixeles
                                        xi, yi, anc, alt = bbox.xmin, bbox.ymin, bbox.width, bbox.height
                                        xi, yi, anc, alt = int( xi * an), int(yi * al), int(anc * an), int(alt * al)

                                        # Offset X
                                        offsetan = (offsetx / 100 ) * anc
                                        xi = int(xi - int(offsetan / 2))
                                        anc = int(anc + offsetan)
                                        xf = xi + anc

                                        # Offset Y
                                        offsetal = (offsety / 100) * alt
                                        yi = int(yi - offsetal)
                                        alt = int(alt + offsetal)
                                        yf = yi + alt

                                        # ERROR

                                        if xi < 0: xi = 0
                                        if y1 < 0: yi = 0
                                        if anc < 0: anc = 0
                                        if alt < 0: alt = 0

                                        # Steps

                                        if step == 0:
                                            # Draw
                                            cv2.rectangle(frame, (xi, yi, anc, alt), (255, 0, 255), 2)

                                            #IMG Step 0
                                            als0, ans0, c = img_step0.shape
                                            frame[50:50 +als0, 50:50 + ans0] = img_step0

                                            # IMG Step 1
                                            als1, ans1, c = img_step1.shape
                                            frame[50:50 + als1, 1030:1030 + ans1] = img_step1

                                            # IMG Step 2
                                            als2, ans2, c = img_step2.shape
                                            frame[270:270 + als2, 1030:1030 + ans2] = img_step2

                                            # Face Center
                                            if x7 > x5 and x8 < x6:
                                                # IMG Check
                                                alch, anch, c = img_check.shape
                                                frame[165:165 + alch, 1105:1105 + anch] = img_check

                                                # Conteo Parpadeo
                                                if longitud1 <= 10 and longitud2 <= 10 and parpadeo == False:
                                                    conteo = conteo + 1
                                                    parpadeo = True

                                                elif longitud2 > 10 and longitud2 > 10 and parpadeo == True:
                                                    parpadeo = False
                                                    # Parpadeos
                                                    # Conteo de parpadeos
                                                cv2.putText(frame, f'Parpadeos: {int(conteo)}', (1070, 375),cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                                                (255, 255, 255), 1)

                                                #Conteo

                                                if conteo >= 3:
                                                    # IMG Check
                                                    alch, anch, c = img_check.shape
                                                    frame[385:385 + alch, 1105:1105 + anch] = img_check

                                                    # Open Eyes
                                                    if longitud1 > 14 and longitud2 > 14:
                                                        # Cut
                                                        cut = frameSave[yi:yf, xi:xf]

                                                        # Save Face
                                                        cv2.imwrite(f"{OutFolderPathFace}/{RegUser}.png", cut)

                                                        # Cerramos
                                                        step = 1

                                            else:
                                                conteo = 0

                                        if step == 1:
                                            # Draw
                                            cv2.rectangle(frame, (xi, yi, an, al), (0, 255, 0), 2)
                                            # IMG check Liveness
                                            allich, anlich, c = img_livCheck.shape
                                            frame[50:50 + allich, 50:50 + anlich] = img_livCheck
                            # Close
                            close = pantalla2.protocol("WM_DELETE_WINDOW", Close_Window)



                            #Circles
                            cv2.circle(frame, (x7,y7), 2, (255,0,0), cv2.FILLED)
                            cv2.circle(frame, (x8, y8), 2, (255, 0, 0), cv2.FILLED)

        # Con Video
        im = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=im)

        # Show Video
        lblVideo.configure(image=img)
        lblVideo.image = img
        lblVideo.after(10, Log_Biometric)

    else:
        cap.release()

# Function Sign
def Sign():
    global LogUser, LogPass, OutFolderPath, cap, lblVideo, pantalla3, FaceCode, clases, images

    # DB Faces
    # Accedemos a la carpeta
    images = []
    clases = []
    lista = os.listdir(OutFolderPathFace)

    # Leemos los rostros del DB
    for lis in lista:
        # Leemos las imagenes de los rostros
        imgdb = cv2.imread(f'{OutFolderPathFace}/{lis}')
        # Almacenamos imagen
        images.append(imgdb)
        # Almacenamos nombre
        clases.append(os.path.splitext(lis)[0])

    # Face Code
    FaceCode = Code_Face(images)

    # 3° Ventana
    pantalla3 = Toplevel(pantalla)
    pantalla3.title("BIOMETRIC SIGN")
    pantalla3.geometry("1280x720")

    back2 = Label(pantalla3, image=imagenB, text="Back")
    back2.place(x=0, y=0, relwidth=1, relheight=1)

    # Video
    lblVideo = Label(pantalla3)
    lblVideo.place(x=0, y=0)

    # Elegimos la camara
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 1280)
    cap.set(4, 720)
    Sign_Biometric()

# Function Log
def Log():
    global RegName, RegUser, RegPass, InputNameReg, InputUserReg, InputPasswordReg, cap, lblVideo, pantalla2
    #Extract Name - User - Pass
    RegName, RegUser, RegPass = InputNameReg.get(), InputUserReg.get(), InputPasswordReg.get()
    # Formulario imcompleto
    if len(RegName) == 0 or len(RegUser) == 0 or len(RegPass) == 0:
        # Info incompleted
        print(" FORMULARIO INCOMPLETO ")

    else:
        # Info Completed
        # Check users
        UserList = os.listdir(PathUserCheck)
        # Name Users
        UserName = []
        for lis in UserList:
            # Extract User
            User = lis
            User = User.split('.')
            # Save
            UserName.append(User[0])

        # Check UserName
        if RegUser in UserName:
            # Registred
            print("USUARIO REGISTRADO ANTERIORMENTE")
        else:
            # No Registred
            # Info
            info.append(RegName)
            info.append(RegUser)
            info.append(RegPass)
            # Export Info
            f = open(f"{OutFolderPathUser}/{RegUser}.txt", "w")
            f.write(RegName + ',')
            f.write(RegUser + ',')
            f.write(RegPass)
            f.close()
            # Clean
            InputNameReg.delete(0, END)
            InputUserReg.delete(0, END)
            InputPasswordReg.delete(0, END)

            #New Screen

            pantalla2 = Toplevel(pantalla)
            pantalla2.title("LOGIN BIOMETRIC")
            pantalla2.geometry("1280x720")

            # Label Video
            lblVideo = Label(pantalla2)
            lblVideo.place(x=0, y=0)

            #Video Captura
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            cap.set(3,1280)
            cap.set(4,720)
            Log_Biometric()


# Umbral
confThresholdCap = 0.5
confThresholdGlass = 0.5



# Path
OutFolderPathUser = 'C:/Proyectos/FacesRecognition/Database/Users'
PathUserCheck = 'C:/Proyectos/FacesRecognition/Database/Users/'
OutFolderPathFace = 'C:/Proyectos/FacesRecognition/Database/Faces'

# Read Images

img_cap = cv2.imread("C:/Proyectos/FacesRecognition/SetUp/cap.png")
img_glass = cv2.imread("C:/Proyectos/FacesRecognition/SetUp/glass.png")
img_check = cv2.imread("C:/Proyectos/FacesRecognition/SetUp/check.png")
img_step0 = cv2.imread("C:/Proyectos/FacesRecognition/SetUp/Step0.png")
img_step1 = cv2.imread("C:/Proyectos/FacesRecognition/SetUp/Step1.png")
img_step2 = cv2.imread("C:/Proyectos/FacesRecognition/SetUp/Step2.png")
img_livCheck = cv2.imread("C:/Proyectos/FacesRecognition/SetUp/LivenessCheck.png")


# Variables
parpadeo = False
conteo = 0
muestra = 0
step = 0
# Offset

offsety = 40
offsetx = 25

# Threshold
confThreshold = 0.5

#Tool Draw
mpDraw = mp.solutions.drawing_utils
ConfigDraw = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

# Object Face Mesh
FacemeshObject = mp.solutions.face_mesh
FaceMesh = FacemeshObject.FaceMesh(max_num_faces=1)

# Object Face Detect
FaceObject = mp.solutions.face_detection
detector = FaceObject.FaceDetection(min_detection_confidence=0.5, model_selection=1)

# infolist
info = []

# Ventana principal
pantalla = Tk()
pantalla.title("FACE RECOGNITION SYSTEM")
pantalla.geometry("1280x720")

# Fondo

imagenF = PhotoImage(file="C:/Proyectos/FacesRecognition/SetUp/Inicio.png")
background = Label(image=imagenF, text="Inicio")
background.place(x=0, y=0, relheight=1, relwidth=1)

# Fondo 2
imagenB = PhotoImage(file="C:/Proyectos/FacesRecognition/SetUp/Back2.png")

# Input Text Login
# Name
InputNameReg = Entry(pantalla)
InputNameReg.place(x=110, y=320)

# User
InputUserReg = Entry(pantalla)
InputUserReg.place(x=110, y=430)

# Pass
InputPasswordReg = Entry(pantalla)
InputPasswordReg.place(x=110, y=540)

# Buttons
# Register
imagenBR = PhotoImage(file="C:/Proyectos/FacesRecognition/SetUp/BtSign.png")
BtReg = Button(pantalla, text="Registro", image=imagenBR, height="40", width="200", command=Log)
BtReg.place(x=300, y=580)

# Login
imagenBL = PhotoImage(file="C:/Proyectos/FacesRecognition/SetUp/BtLogin.png")
BtSign = Button(pantalla, text="Registro", image=imagenBL, height="40", width="200", command=Sign)
BtSign.place(x=900, y=580)

pantalla.mainloop()
