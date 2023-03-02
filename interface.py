import cv2
import numpy as np
import time
drawing = False # true if mouse is pressed
pt1_x , pt1_y = None , None
thick = 20

# mouse callback function
def line_drawing(event,x,y,flags,param):
    global pt1_x,pt1_y,drawing, thick

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        pt1_x,pt1_y=x,y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=thick)
            pt1_x,pt1_y=x,y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=thick)


img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('test draw')
cv2.setMouseCallback('test draw',line_drawing)

while(1):
    t = time.localtime()
    cv2.imshow('test draw',img)
    key = cv2.waitKey(1)
    if key == ord('s'):
        cv2.imwrite(f"submodules/3DHumanGeneration/train/0422_graphAE_dfaust/diffusion/doodle_images/{str(time.strftime('%H_%M_%S', t))}.png", img)
        break
    if key == ord('-'):
        thick -= 1
    elif key == ord('+'):
        thick += 1
    if key == ord('k'):
        break

cv2.destroyAllWindows()