import cv2
import numpy as np

image = cv2.imread("sil.png", 0)
edges = cv2.Sobel(image, cv2.CV_64F, 1, 1)
edges = edges.astype(np.uint8)

ret, thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
white_only = np.zeros_like(image)
white_only[thresh == 255] = 255
noise = cv2.blur(white_only, (6, 6))
result = cv2.add(edges, noise, dtype=cv2.CV_64F)


cv2.imshow('Result', result)
scale_factor = np.iinfo(np.uint16).max / np.amax(result)
result = np.uint16(result * scale_factor)
cv2.imwrite("experiments/ShapeModel_test/001/3blurred.png", result)
cv2.imwrite("silblurred.png", np.float32(result))
cv2.waitKey(0)
cv2.destroyAllWindows()

