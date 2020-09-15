import numpy as np
from skimage import io, img_as_float, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio
from cv2 import cv2
from math import log10, sqrt 
from sklearn.metrics import mean_squared_error

# Add ruido branco com media = mean e desvio padrao = std_dev na imagem passada como parametro
def add_gaussian_noise2(image, mean=0, std_dev=1):
    gauss = np.random.normal(mean,std_dev,image.size)
    gauss = gauss.reshape(image.shape[0], image.shape[1]).astype('uint8')

    img_gauss = cv2.add(image, gauss)
    return img_gauss

### MAIN ### 

# Abre a imagem e converte para RGB
img = cv2.imread('fig1.jpg', cv2.COLOR_BGR2RGB)

# Conversão para tons de cinza
img_grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

cv2.imshow('img grayscale', img_grayscale)
cv2.waitKey()

img_awgn = add_gaussian_noise2(img_grayscale)
io.imsave('img_awgn.jpg', img_awgn)

cv2.imshow('img awgn', img_awgn)
cv2.waitKey()

# Filtro NLM
nlm_denoised = cv2.fastNlMeansDenoising(img_awgn, h=52)
io.imsave('nlm_h_52.jpg', nlm_denoised)

nlm_denoised2 = cv2.fastNlMeansDenoising(img_awgn, h=9)
io.imsave('nlm_h_9.jpg', nlm_denoised2)

print("PSNR NLM = " + str(cv2.PSNR(img_grayscale, nlm_denoised)) + " dB")
print("MSE      = " + str(mean_squared_error(img_grayscale, nlm_denoised2)))
print("-----------------------------------------------")

# Filtro bilateral
bf_denoised = cv2.bilateralFilter(img_awgn, 5, 10, 10)
print("PSNR BF  = " + str(cv2.PSNR(img_grayscale, bf_denoised)) + " dB")
print("MSE      = " + str(mean_squared_error(img_grayscale, bf_denoised)))
io.imsave('bf_10.jpg', bf_denoised)

bf_denoised2 = cv2.bilateralFilter(img_awgn, 5, 75, 75)
print("PSNR BF2 = " + str(cv2.PSNR(img_grayscale, bf_denoised2)) + " dB")
print("MSE      = " + str(mean_squared_error(img_grayscale, bf_denoised2)))
io.imsave('bf_75.jpg', bf_denoised2)

bf_denoised3 = cv2.bilateralFilter(img_awgn, 5, 150, 150)
print("PSNR BF3 = " + str(cv2.PSNR(img_grayscale, bf_denoised3)) + " dB")
print("MSE      = " + str(mean_squared_error(img_grayscale, bf_denoised3)))
io.imsave('bf_150.jpg', bf_denoised3)
print("-----------------------------------------------")

# Filtro de mediana
kernel_size = 5
median = cv2.medianBlur(img_awgn, 3)
print("PSNR MF  = " + str(cv2.PSNR(img_grayscale, median)) + " dB")
print("MSE      = " + str(mean_squared_error(img_grayscale, median)))
io.imsave('mf_3.jpg', median)

median2 = cv2.medianBlur(img_awgn, 5)
print("PSNR MF2 = " + str(cv2.PSNR(img_grayscale, median2)) + " dB")
print("MSE      = " + str(mean_squared_error(img_grayscale, median2)))
io.imsave('mf_5.jpg', median2)

median3 = cv2.medianBlur(img_awgn, 7)
print("PSNR MF3 = " + str(cv2.PSNR(img_grayscale, median3)) + " dB")
print("MSE      = " + str(mean_squared_error(img_grayscale, median3)))
io.imsave('mf_7.jpg', median3)
