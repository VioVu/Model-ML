from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import os
import generate  # Import mô-đun generate.py của bạn


def choose_file():
    """
    Cho phép người dùng chọn ảnh từ file và hiển thị trong giao diện.
    """
    filename = filedialog.askopenfilename(
        initialdir=os.getcwd(),
        title="Select image file",
        filetypes=(("JPG File", "*.jpg"), ("PNG file", "*.png"), ("All files", "*.*"))
    )
    
    if filename:
        # Cập nhật entry với đường dẫn ảnh
        entry1.delete(0, 'end')
        entry1.insert(0, filename)

        # Hiển thị ảnh trong giao diện
        img = Image.open(filename)
        img.thumbnail((550, 400))  # Resize ảnh vừa đủ
        img = ImageTk.PhotoImage(img)

        # Cập nhật ảnh hiển thị
        lbl.configure(image=img)
        lbl.image = img


def generate_caption():
    """
    Gọi hàm run_model() để dự đoán caption từ ảnh đã chọn và hiển thị kết quả.
    """
    file_path = entry1.get()
    
    if not file_path:
        result_label.config(text="Please select an image first.")
        return

    if not os.path.isfile(file_path):  # Kiểm tra đường dẫn hợp lệ
        result_label.config(text="Invalid file path.")
        return

    try:
        # Gọi hàm run_model() và nhận kết quả
        caption = generate.run_model(file_path)
        
        if caption:
            result_label.config(text=f"Caption: {caption}")
        else:
            result_label.config(text="Could not generate caption.")
    except Exception as e:
        result_label.config(text=f"Error: {str(e)}")


# Tạo giao diện chính
root = Tk()
root.title("Image Caption Generator")
root.geometry("700x700")

# Hiển thị ảnh đã chọn
lbl = Label(root)
lbl.pack(pady=10)

# Khung nhập liệu và các nút điều khiển
frm = Frame(root)
frm.pack(side=BOTTOM, padx=10, pady=10)

# Entry để hiển thị đường dẫn ảnh
entry1 = Entry(frm, width=60)
entry1.pack(pady=5)

# Thêm các nút điều khiển
button1 = Button(frm, text="Select Image", command=choose_file, width=20)
button1.pack(pady=5)

button2 = Button(frm, text="Generate Caption", command=generate_caption, width=20)
button2.pack(pady=5)

# Label để hiển thị caption hoặc thông báo
result_label = Label(root, text="", wraplength=600, font=("Helvetica", 12), justify=LEFT)
result_label.pack(pady=10)

# Khởi chạy giao diện
root.mainloop()
