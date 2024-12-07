import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tải mô hình đã huấn luyện
model_path = r"D:\Kì 1 2024-2025\Machine Learning\Model Final\Finallll\model_new.keras"
wordtoix_path = r"D:\Kì 1 2024-2025\Machine Learning\Model Final\Finallll\wordtoix.npy"
ixtoword_path = r"D:\Kì 1 2024-2025\Machine Learning\Model Final\Finallll\ixtoword.npy"

model = load_model(model_path)
wordtoix = np.load(wordtoix_path, allow_pickle=True).item()
ixtoword = np.load(ixtoword_path, allow_pickle=True).item()

# Thiết lập tham số
max_length = 70

# Tải mô hình InceptionV3 để mã hóa ảnh
inception_model = InceptionV3(weights='imagenet')
model_new = Model(inception_model.input, inception_model.layers[-2].output)

def preprocess(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def encode_image(image_path):
    img = preprocess(image_path)
    feature_vector = model_new.predict(img, verbose=0)
    feature_vector = feature_vector.reshape((1, feature_vector.shape[1]))
    return feature_vector

def greedySearch(photo):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    return ' '.join(final)

def predict_caption(image):
    encoded_img = encode_image(image)
    caption = greedySearch(encoded_img)
    return caption

# Tạo giao diện Gradio
iface = gr.Interface(fn=predict_caption, inputs=gr.Image(type="filepath"), outputs=gr.Text())
iface.launch()
share=True
private=False