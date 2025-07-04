import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
# Main window configuration
window = tk.Tk()
window.title("Handwritten Digit Recognition System")
window.geometry("800x500")
window.configure(bg="#f4f4f4")

# Header
header_frame = tk.Frame(window, bg="#2b3e50", height=80)
header_frame.pack(fill=tk.X)
header_label = tk.Label(header_frame, text="Handwritten Digit Recognition", font=('Helvetica', 24, 'bold'), fg="white", bg="#2b3e50")
header_label.pack(pady=20)

# Main content frame
content_frame = tk.Frame(window, bg="#f4f4f4", padx=20, pady=20)
content_frame.pack(fill=tk.BOTH, expand=True)

# Digit label and entry
l1 = tk.Label(content_frame, text="Digit:", font=('Helvetica', 16), bg="#f4f4f4")
l1.grid(row=0, column=0, sticky="w", pady=10, padx=10)
t1 = ttk.Entry(content_frame, width=20, font=('Helvetica', 14))
t1.grid(row=0, column=1, pady=10, padx=10)

# Screen capture function
def screen_capture():
    import pyscreenshot as ImageGrab
    import time
    import os
    os.startfile(r"C:\Users\urvas\OneDrive\Desktop\mspaint.lnk")
    s1 = t1.get()
    os.chdir("E:/DS and ML/Untitled Folder/Untitled Folder/captured_images")
    os.mkdir(s1)
    os.chdir("E:/DS and ML/Untitled Folder/Untitled Folder/")

    images_folder = "captured_images/" + s1 + "/"
    time.sleep(15)
    for i in range(0, 5):
        time.sleep(8)
        im = ImageGrab.grab(bbox=(150,300,700,750))  # x1, y1, x2, y2
        print("saved......", i)
        im.save(images_folder + str(i) + '.png')
        print("clear screen now and redraw now........")
    messagebox.showinfo("Result", "Capturing screen is completed!!")

# Generate dataset function
def generate_dataset():
    import cv2
    import csv
    import glob

    header = ["label"]
    for i in range(0, 784):
        header.append("pixel" + str(i))
    with open('dataset.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    for label in range(10):
        dirList = glob.glob("captured_images/" + str(label) + "/*.png")

        for img_path in dirList:
            im = cv2.imread(img_path)
            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im_gray = cv2.GaussianBlur(im_gray, (15, 15), 0)
            roi = cv2.resize(im_gray, (28, 28), interpolation=cv2.INTER_AREA)

            data = []
            data.append(label)
            rows, cols = roi.shape

            for i in range(rows):
                for j in range(cols):
                    k = roi[i, j]
                    if k > 100:
                        k = 1
                    else:
                        k = 0
                    data.append(k)
            with open('dataset.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(data)
    messagebox.showinfo("Result", "Generating dataset is completed!!")

# Train and save accuracy function
def train_save_accuracy():
    import pandas as pd
    from sklearn.utils import shuffle
    data = pd.read_csv('dataset.csv')
    data = shuffle(data)
    X = data.drop(["label"], axis=1)
    Y = data["label"]
    from sklearn.model_selection import train_test_split
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)
    import joblib
    from sklearn.svm import SVC
    classifier = SVC(kernel="linear", random_state=6)
    classifier.fit(train_x, train_y)
    joblib.dump(classifier, "model/digit_recognizer")
    from sklearn import metrics
    prediction = classifier.predict(test_x)
    acc = metrics.accuracy_score(prediction, test_y)
    messagebox.showinfo("Result", f"Your accuracy is {acc}")

# Prediction function
def prediction():
    import joblib
    import cv2
    import numpy as np
    import time
    import pyscreenshot as ImageGrab
    import os
    os.startfile(r"C:\Users\urvas\OneDrive\Desktop\mspaint.lnk")
    model = joblib.load("model/digit_recognizer")
    images_folder = "img/"
    time.sleep(15)
    while True:
        img = ImageGrab.grab(bbox=(150,300,700,750))

        img.save(images_folder + "img.png")
        im = cv2.imread(images_folder + "img.png")
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_gray = cv2.GaussianBlur(im_gray, (15, 15), 0)

        ret, im_th = cv2.threshold(im_gray, 100, 255, cv2.THRESH_BINARY)
        roi = cv2.resize(im_th, (28, 28), interpolation=cv2.INTER_AREA)

        rows, cols = roi.shape

        X = []

        for i in range(rows):
            for j in range(cols):
                k = roi[i, j]
                if k > 100:
                    k = 1
                else:
                    k = 0
                X.append(k)

        predictions = model.predict([X])
        print("Prediction:", predictions[0])

        cv2.putText(im, "Prediction is: " + str(predictions[0]), (20, 20), 0, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.startWindowThread()
        cv2.namedWindow("Result")
        cv2.imshow("Result", im)
        cv2.waitKey(10000)
        if cv2.waitKey(1) == 13:
            break
    cv2.destroyAllWindows()

# Buttons
button_style = {'font': ('Helvetica', 14), 'bg': '#007acc', 'fg': 'white', 'relief': 'groove', 'borderwidth': 2, 'padx': 10, 'pady': 5}

b1 = tk.Button(content_frame, text="1. Open paint and capture the screen", command=screen_capture, **button_style)
b1.grid(row=1, column=0, columnspan=2, pady=10, sticky="ew")

b2 = tk.Button(content_frame, text="2. Generate dataset", command=generate_dataset, **button_style)
b2.grid(row=2, column=0, columnspan=2, pady=10, sticky="ew")

b3 = tk.Button(content_frame, text="3. Train the model, save it and calculate accuracy", command=train_save_accuracy, **button_style)
b3.grid(row=3, column=0, columnspan=2, pady=10, sticky="ew")

b4 = tk.Button(content_frame, text="4. Live prediction", command=prediction, **button_style)
b4.grid(row=4, column=0, columnspan=2, pady=10, sticky="ew")

window.mainloop()
