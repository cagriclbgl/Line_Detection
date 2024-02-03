import numpy as np
import cv2 as cv
import os

def tespit_ve_kaydet(girdi_klasoru, cikti_klasoru):
    # Eğer çıkış klasörü yoksa oluştur
    if not os.path.exists(cikti_klasoru):
        os.makedirs(cikti_klasoru)

    # Girdi klasöründeki dosyaları kontrol et
    dosya_listesi = os.listdir(girdi_klasoru)

    for dosya in dosya_listesi:
        if dosya.endswith(".jpg") or dosya.endswith(".jpeg"):
            # Resmi oku
            dosya_yolu = os.path.join(girdi_klasoru, dosya)
            img = cv.imread(dosya_yolu)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            gray = np.float32(gray)
            dst = cv.cornerHarris(gray, 10, 3, 0.04)
            dst = cv.dilate(dst, None)
            img[dst > 0.01 * dst.max()] = [0, 0, 255]

            # Çıktı klasörüne işlenmiş resmi kaydet
            cikti_yolu = os.path.join(cikti_klasoru, 'islenmis_' + dosya)
            resized = cv.resize(img, (640, 480))
            cv.imwrite(cikti_yolu, resized)

            print(f"{dosya} işlendi ve {cikti_yolu} konumuna kaydedildi.")

# İki farklı girdi ve çıktı klasörü tanımla
tepe_girdi_klasoru = r"C:\Users\berka\PycharmProjects\pythonProject\data_set\tepeden"
tepe_cikti_klasoru = r"C:\Users\berka\PycharmProjects\pythonProject\output\tepe"

yan_girdi_klasoru = r"C:\Users\berka\PycharmProjects\pythonProject\data_set\yandan"
yan_cikti_klasoru = r"C:\Users\berka\PycharmProjects\pythonProject\output\yan"

# Tepeden ve yandan gelen görüntüleri sırasıyla işle ve çıktıya kaydet
tespit_ve_kaydet(tepe_girdi_klasoru, tepe_cikti_klasoru)
tespit_ve_kaydet(yan_girdi_klasoru, yan_cikti_klasoru)