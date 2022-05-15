import numpy as np
import cv2
import os
import random

from PIL import ImageFont, ImageDraw, Image
from matplotlib.image import imread
from sklearn.metrics import accuracy_score
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

# đường dẫn tới folder training
train_path = "C:/Users/n/Desktop/fruit-classify/fruits/Training/"
test_path = "C:/Users/n/Desktop/fruit-classify/fruits/Test/"

#categories_test = [name for name in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, name))]

# lấy tên của tất cả folder có bên trong train_path
categories_train = [name for name in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, name))]
#categories_train = ['Táo', 'Bơ', 'Chuối', 'Dưa lưới', 'Khế', 'Quả anh đào', 'Quả sung', 'Chanh dây', 'Nho', 'Ổi', 'Quả hồng', 
#'Quả dương đào', 'Vải', 'Xoài', 'Măng cụt', 'Dưa hấu', 'Cam', 'Đu đủ', 'Đào', 'Lê', 'Thanh long', 'Trái lựu', 'Chôm chôm', 'Quả dâu tây']

# load model đã tạo và chạy trong main.py
model = load_model('model.h5')

# mở webcam trên laptop
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# đổi màu cho text
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (0, 255, 0)
fontcolor1 = (0, 0, 255)
font = ImageFont.truetype("arial.ttf", 18)

while True:
    # đọc từng ảnh có trong video
    ret, frame = video.read()

    # kiểm tra nếu ảnh là khác rỗng
    if frame is not None:
        frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)

        # chỉnh sửa kích cỡ của ảnh về 100, 100 để phù hợp với model đã load bên trên
        resize_frame = cv2.resize(frame, (100, 100))
        predict = model.predict(np.expand_dims(img_to_array(resize_frame), 0))

        # lấy tên của vật đã nhận dạng trong categories_train
        predict_string = categories_train[np.argmax(predict)]

        # tạo text phía trên video
        #draw = ImageDraw.Draw(Image.fromarray(frame))
        #draw.text((30,30), predict_string, font=font, fill=fontcolor)
        cv2.putText(frame, predict_string, (0,0+0+30), fontface, fontscale, fontcolor, 2)

        # hiển thị 1 cửa sổ tên là 'frames' với ảnh frame
        cv2.imshow('frames', frame)
    
    # chờ 100 ms và kiểm tra nếu bấm q thì sẽ dừng chương trình
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# giải phóng video để tránh memory leak
video.release()

# đóng tất cả cửa sổ
cv2.destroyAllWindows()