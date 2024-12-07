import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Đọc các file từ điển word_to_index và index_to_word
print("Loading word-index mappings...")
with open(r"D:\Kì 1 2024-2025\Machine Learning\Model Final\Finallll\wordtoix.pkl", 'rb') as file:
    word_to_index = pickle.load(file)

with open(r"D:\Kì 1 2024-2025\Machine Learning\Model Final\Finallll\ixtoword.pkl", 'rb') as file:
    index_to_word = pickle.load(file)

# Tải mô hình sinh caption
print("Loading the caption model...")
model = load_model(r"D:\Kì 1 2024-2025\Machine Learning\Model Final\Finallll\model_new.keras")

# Tải mô hình Inception v3 để trích xuất đặc trưng ảnh
print("Loading the Inception V3 model...")
inception_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

def preprocess_image(img_path):
    """
    Tiền xử lý ảnh để phù hợp với Inception v3.
    :param img_path: Đường dẫn đến ảnh
    :return: Feature vector của ảnh
    """
    # Resize ảnh về 299x299
    img = load_img(img_path, target_size=(299, 299))
    img = img_to_array(img)

    # Chuẩn hóa đầu vào
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def encode_image(img_path):
    """
    Mã hóa ảnh thành vector đặc trưng bằng mô hình Inception v3.
    :param img_path: Đường dẫn ảnh
    :return: Vector đặc trưng (feature vector) của ảnh
    """
    img_array = preprocess_image(img_path)
    # Trích xuất đặc trưng ảnh
    feature_vector = inception_model.predict(img_array)
    return feature_vector

def predict_caption(photo, max_length=70):
    """
    Sinh caption từ ảnh đầu vào (photo features) bằng mô hình đã huấn luyện.
    :param photo: Đặc trưng ảnh từ Inception v3 (1, 2048)
    :param max_length: Độ dài tối đa của chuỗi đầu vào
    :return: Caption được sinh ra
    """
    inp_text = "startseq"
    word_count = {}  # Đếm số lần xuất hiện của mỗi từ

    for _ in range(max_length):  # Lặp cho đến khi đạt max_length
        # Chuyển từ chuỗi hiện tại sang chỉ số
        sequence = [word_to_index.get(word, 0) for word in inp_text.split()]
        # Pad sequence để đảm bảo kích thước đúng max_length
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post', truncating='post')

        # Dự đoán từ tiếp theo
        ypred = model.predict([photo, sequence], verbose=0)
        ypred = ypred.argmax()

        # Lấy từ tương ứng từ chỉ số
        word = index_to_word.get(ypred, "")
        if not word:  # Nếu không tìm thấy từ, thoát vòng lặp
            break

        # Kiểm tra số lần xuất hiện của từ
        if word in word_count:
            word_count[word] += 1
            if word_count[word] > 3:
                break  # Dừng vòng lặp nếu từ xuất hiện quá 3 lần
        else:
            word_count[word] = 1

        inp_text += " " + word
        if word == "endseq":  # Kết thúc khi gặp 'endseq'
            break

    # Loại bỏ "startseq" và "endseq" khỏi caption cuối
    final_caption = " ".join(inp_text.split()[1:-1])
    return final_caption


def run_model(img_path):
    """
    Chạy toàn bộ pipeline: Tiền xử lý, mã hóa ảnh, sinh caption và hiển thị kết quả.
    """
    try:
        # Tiền xử lý và mã hóa ảnh
        print("Bắt đầu tiền xử lý và mã hóa ảnh...")
        photo_features = encode_image(img_path)  # Vector đặc trưng từ Inception v3

        # Sinh caption
        print("Sinh caption từ mô hình...")
        caption = predict_caption(photo_features)

        # Hiển thị ảnh và caption
        img_data = plt.imread(img_path)
        plt.imshow(img_data)
        plt.axis("off")
        plt.title(f"Caption: {caption}", fontsize=12, color="blue")
        plt.show()

        return caption
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")
        return None




