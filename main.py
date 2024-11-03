#Nota do autor: A versão atual de Python há incompatibilidade com novas versões do Tenserflow e vice-versa, então para que o programa funcione será necessário retroceder a versão de ambos. Utilizei o Python 3.10 e o Tensorflow na versão 2.13.0


#Foram gastas 1 térmica de café até agora

#Espero que esteja tendo um bom dia. :)

#Sandro, não julgue, eu gosto de IA preditiva.

#Alef, eu te amo e você é a melhor companhia que alguém poderia ter. Sou um homem de sorte. <3




from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
import numpy as np
import cv2
import time


# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Defina os objetos personalizados
custom_objects = {
    'DepthwiseConv2D': DepthwiseConv2D
}

# Carregue o modelo com os objetos personalizados
model = load_model("keras_model.h5", custom_objects=custom_objects, compile=False)
model.save("saved_model/")
model = load_model("saved_model")
# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
    time.sleep(2)

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
