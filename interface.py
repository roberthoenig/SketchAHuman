import cv2
import numpy as np
import time
import copy

drawing = False # true if mouse is pressed
pt1_x , pt1_y = None , None
thick = 20
color = (255, 255, 255)
deleting = False


# mouse callback function
def mouse_callback(event, x, y, flags, params):
    global drawing, deleting, pt1_x, pt1_y, thick, color, img, cache

    if event==cv2.EVENT_LBUTTONDOWN:
        if not deleting:
            drawing=True
            pt1_x,pt1_y=x,y
            cache = copy.deepcopy(img)

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True and deleting == False:
            cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=thick)
            pt1_x,pt1_y=x,y

    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        if deleting == False:
            cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=thick)
        elif deleting == True:
            img = copy.deepcopy(cache)
            cv2.imshow('sketch', img)




img = np.zeros((512,512,3), np.uint8)
cache = copy.deepcopy(img)
cv2.namedWindow('sketch')
cv2.setMouseCallback('sketch', mouse_callback)


while True:
    t = time.localtime()
    cv2.imshow('sketch', img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        cv2.imwrite(
            f"submodules/3DHumanGeneration/train/0422_graphAE_dfaust/diffusion/doodle_images/{str(time.strftime('%H_%M_%S', t))}.png",
            img)
        break
    elif k == ord('d'):
        # switch to deleting mode
        deleting = True
        drawing = False
        print('Deleting mode')
    elif k == ord('w'):
        deleting = False
        print("Writing mode")
    elif k == ord('n'):
        cv2.imwrite(
            f"submodules/3DHumanGeneration/train/0422_graphAE_dfaust/diffusion/doodle_images/{str(time.strftime('%H_%M_%S', t))}.png",
            img)
        img = np.zeros((512, 512, 3), np.uint8)
    elif k == ord('c'):
        # clear the image and start over
        img = np.zeros((512, 512, 3), np.uint8)
        lines.clear()
        contours.clear()

cv2.destroyAllWindows()