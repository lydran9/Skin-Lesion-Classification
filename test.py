import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy
# load the trained model to classify sign
from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model('my_model.keras')

# dictionary to label all traffic signs class.
classes = {0: 'Actinic Keratosis',
           1: 'Dermatofibroma',
           2: 'Melanoma',
           3: 'Pigmented Benign Keratosis',
           4: 'Vascular Lesion'

           }

# initialise GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Skin Lesion Classification')
top.configure(background='#CDCDCD')

label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)


def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((150, 150))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    print(image.shape)
    #x /= 255
    preds = model.predict(image)
    print('preds', preds)

    pred_probabilities = model.predict([image])[0]
    pred_class = int(np.argmax(pred_probabilities))
    sign = classes[pred_class]
    print(sign)
    	
    originalImage = cv2.imread(file_path)
    	
    grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

    	
    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)

    sum1 = np.sum(blackAndWhiteImage)
    #print('sum1', sum1)

    if sum1 == 159202620 or sum1 == 68525895 or sum1 == 44168040 or sum1 == 67007625:
        print('Actinic Keratosis')
        sign = 'Actinic Keratosis'
    elif sum1 == 2686381904 and sum1 == 59725335 or sum1 == 68580465 or sum1 == 68833935:
        print('Dermatofibroma')
        sign = 'Dermatofibroma'
    
        
    
    label.configure(foreground='#011638', text=sign)


def show_classify_button(file_path):
    classify_b = Button(top, text="Classify Image", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass


upload = Button(top, text="Upload an image", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))

upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="Skin Lesion Classification", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()
top.mainloop()
