import cv2
import numpy as np
import os


def process_image(img):
    # Görüntüyü yeniden boyutlandır
    img = cv2.resize(img, (640, 480))

    # Griye çevir
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Eşik değer uygula
    blackAndWhiteImage2 = cv2.threshold(gray_image, 80, 255, cv2.THRESH_BINARY)[1]

    # Median blur filtresi uygula
    kernelSize = 5
    medianBlurredImage = cv2.medianBlur(img, kernelSize)

    # Kontrastı artır
    contrast_factor = 1.5
    increased_contrast_image = cv2.convertScaleAbs(medianBlurredImage, alpha=contrast_factor, beta=0)

    # Canny kenar dedektörünü uygula
    edges_canny = cv2.Canny(medianBlurredImage, 20, 90)

    # Hough çizgilerini bul
    lines = cv2.HoughLinesP(edges_canny, 1, np.pi / 180, 1, minLineLength=0, maxLineGap=2)
    img_with_lines = img.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Konturları bul ve çiz
    contours, hierarchy = cv2.findContours(edges_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_with_lines, (x, y), (x + w, y + h), (0, 255, 0), 1)

    return img_with_lines


def process_directory(input_dir, output_dir):
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Görüntü yüklenemedi: {img_path}")
            continue

        processed_image = process_image(img)

        output_img_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_img_path, processed_image)


# Ana dosya yollarını tanımla
dataset_path = "data_set"  # Görüntülerin olduğu ana klasör
output_path = "output"  # İşlenmiş görüntülerin kaydedileceği ana klasör

# 'Tepeden' ve 'Yandan' klasörleri için işlem
for category in ["tepeden", "yandan"]:
    input_dir = os.path.join(dataset_path, category)
    output_dir = os.path.join(output_path, f"{category}_output")

    # Eğer çıktı klasörü yoksa oluştur
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    process_directory(input_dir, output_dir)

# Tüm pencereleri kapat
cv2.destroyAllWindows()

