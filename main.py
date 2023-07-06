import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import warp_polar

def rotate_image(image, angle):
  ''' поворот изображение без отсечения '''
  h, w = image.shape[:2]
  # left = int((np.sqrt(h ** 2 + w ** 2) - w) / 2)
  # image = cv.copyMakeBorder(image, left, left, left, left, cv.BORDER_CONSTANT, None, (0,0,0))

  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)

  cos = np.abs(rot_mat[0, 0])
  sin = np.abs(rot_mat[0, 1])
  # compute the new bounding dimensions of the image
  nW = int((h * sin) + (w * cos))
  nH = int((h * cos) + (w * sin))
  # adjust the rotation matrix to take into account translation
  rot_mat[0, 2] += (nW / 2) - image_center[0]
  rot_mat[1, 2] += (nH / 2) - image_center[1]

  result = cv.warpAffine(image, rot_mat, (nW, nH), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)

  return result


def deskew_view(img, trim = True):
  '''
  text deskewing for OCR using a FFT with visualisation
  :param img: img as openCV np.array
  :param trim: whether to trim fields leaving center of image only
  :return: angle to deskew img
  '''

  ACCURACY = 0.5
  orig = img

  if trim:
    h, w = img.shape[:2]
    delta_h = int(h * 0.3)
    delta_w = int(w * 0.3)
    img = img[delta_h: h - delta_h, delta_w: w - delta_w]         # отбрасываю поля

  bw = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  binary = cv.threshold(bw, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

  dft = cv.dft(np.float32(binary), flags=cv.DFT_COMPLEX_OUTPUT)
  dft_shift = np.fft.fftshift(dft)
  magnitude_spectrum = 20 * np.log(cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

  shape = magnitude_spectrum.shape
  radius = shape[0] // 2 - 1
  warped_image_fs = warp_polar(magnitude_spectrum, radius=radius, output_shape= (360/ACCURACY, 360),
                              scaling = 'linear', order=0) # точность - 0,5 градуса
  warped_image_fs = warped_image_fs[: int(180/ACCURACY), :]  # only use half of FFT

  vals = np.sum(warped_image_fs, axis = 1)

  sample_angles = np.linspace(0, 181, num = int(180/ACCURACY))

  fig = plt.figure(figsize=(10, 10))
  ax1 = fig.add_subplot(1,2,1)
  ax1.imshow(magnitude_spectrum, cmap='magma')
  ax1.title.set_text('FFT of image')

  ax2 = fig.add_subplot(1,2,2)
  ax2.imshow(warped_image_fs, cmap='magma')
  ax2.title.set_text('warped_image')
  ax2.axhline(180, alpha = 0.5)
  plt.show()

  plt.plot(sample_angles, vals)
  plt.axvline(90, alpha = 0.3)
  plt.xticks(np.arange(0,181,10), rotation=45)
  plt.grid()
  plt.show()

  vals = vals[int(45/ACCURACY): int(135/ACCURACY)]                      # выбираем углы от 45 до 135
  rot_angle = 90 - (np.argmax(vals) / int(1/ACCURACY) + 45)
  rotated_img = rotate_image(orig, -rot_angle)

  print('angle: ', rot_angle)
  h_dif = rotated_img.shape[0] - img.shape[0]
  img_big = cv.copyMakeBorder(img, 0, h_dif, 0, 0 , cv.BORDER_CONSTANT, value=(255, 255, 255))
  img_view = np.hstack((img_big, rotated_img))
  img_view = cv.cvtColor(img_view, cv.COLOR_BGR2RGB)
  plt.imshow(img_view)


  return rotated_img, rot_angle
  
  
def deskew(img: np.array, trim = True) -> float:
  '''
  text deskewing for OCR using a FFT
  :param img: img as openCV np.array
  :param trim: whether to trim fields leaving center of image only
  :return: angle to deskew img
  '''

  ACCURACY = 0.5

  if trim:
    h, w = img.shape[:2]
    delta_h = int(h * 0.3)
    delta_w = int(w * 0.3)
    img = img[delta_h: h - delta_h, delta_w: w - delta_w]         # отбрасываю поля

  bw = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  binary = cv.threshold(bw, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

  dft = cv.dft(np.float32(binary), flags=cv.DFT_COMPLEX_OUTPUT)
  dft_shift = np.fft.fftshift(dft)
  magnitude_spectrum = 20 * np.log(cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

  shape = magnitude_spectrum.shape
  radius = shape[0] // 2 - 1
  warped_image_fs = warp_polar(magnitude_spectrum, radius=radius, output_shape= (360/ACCURACY, 360),
                              scaling = 'linear', order=0) # точность - 0,5 градуса
  warped_image_fs = warped_image_fs[: int(180/ACCURACY), :]  # only use half of FFT

  vals = np.sum(warped_image_fs, axis = 1)

  vals = vals[int(45/ACCURACY): int(135/ACCURACY)]                      # angles from 45 to 135
  rot_angle = 90 - (np.argmax(vals) / int(1/ACCURACY) + 45)

  return rot_angle
  
