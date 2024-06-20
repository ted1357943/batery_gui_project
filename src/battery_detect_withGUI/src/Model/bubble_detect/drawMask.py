import cv2 as cv
import numpy as np
from collections import deque
from datetime import datetime
import os

center = None
class maskCreate:
    def __init__(self, windowName, inputImg, mode):
        self.mode = mode
        self.img_tmp = deque([])
        self.mask_tmp = deque([])
        self.img = inputImg.copy()
        self.mask = 255 * np.ones(inputImg.shape, np.uint8)
        self.canvas_mask = self.mask.copy()
        self.canvas = inputImg.copy()
        self.windowName = windowName
        self.createWindow()
        self.setMouseCB()

    def createWindow(self):
        cv.namedWindow(self.windowName)

    def draw_circle(self, event, x, y, flags, param):
        global center
        if event == cv.EVENT_LBUTTONDOWN:
            # print("Left button down")
            # Set center point when left mouse button is clicked
            center = (x, y)
            self.img_tmp.append(self.img.copy())
            self.mask_tmp.append(self.mask.copy())
            if len(self.img_tmp) > 50:
                print("The length of img_tmp is too long")
                print("pop left element, current length is {}".format(len(self.img_tmp)))
                self.img_tmp.popleft()   
                self.mask_tmp.popleft() 

        elif event == cv.EVENT_LBUTTONUP:
            # print("Left button up")
            # Calculate radius and draw circle when left mouse button is released
            if self.mode == 0:
                radius = 5
            else:
                radius = int(((x-center[0])**2 + (y-center[1])**2)**0.5)
            cv.circle(self.img, center, radius, (0,0,0), -1)
            cv.circle(self.mask, center, radius, (0,0,0), -1)
            np.copyto(self.canvas, self.img)
            np.copyto(self.canvas_mask, self.mask)
            cv.imshow(self.windowName, self.canvas)

        elif event==cv.EVENT_MOUSEMOVE and flags ==cv.EVENT_FLAG_LBUTTON:
            # print("Mouse Drag with left botton")
            # Draw the circle's range when draw mouse's left button
            np.copyto(self.canvas, self.img)
            np.copyto(self.canvas_mask, self.mask)
            radius = int(((x-center[0])**2 + (y-center[1])**2)**0.5)
            cv.circle(self.canvas, center, radius, (0, 0, 255), thickness= 4)
            cv.imshow(self.windowName, self.canvas)

        elif event==cv.EVENT_MBUTTONDBLCLK and len(self.img_tmp) != 0:
            # print("Middle botton double click")
            # Undo last change
            print("Length of img_tmp: {}".format(len(self.img_tmp)))
            np.copyto(self.canvas, self.img_tmp.pop())
            np.copyto(self.canvas_mask, self.mask_tmp.pop())
            np.copyto(self.img, self.canvas)
            np.copyto(self.mask, self.canvas_mask)
            cv.imshow(self.windowName, self.canvas)

    def setMouseCB(self):
        cv.setMouseCallback(self.windowName, self.draw_circle)

    def showCanvas(self):
        # Initial: Draw Canvas 
        cv.imshow(self.windowName, self.canvas)

    def createMask(self):
        self.showCanvas()
        while True:
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        self.saveMask()
        cv.destroyAllWindows()
        return self.canvas_mask
    
    def saveMask(self):
        # Get current date and time as a string
        now = datetime.now()
        timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')
        timestampForDir = now.strftime('%Y-%m-%d')

        # Set the output filename with the current date
        output_dir = f'../data/created_mask/{timestampForDir}'
        outputFilename =  os.path.join(output_dir, f'{self.windowName}_{timestamp}.png')
        print("Save {} as {}".format(self.windowName, outputFilename))

        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the image
        cv.imwrite(outputFilename, self.canvas_mask)

    def __del__(self):
        cv.destroyAllWindows()

if __name__ == '__main__':
    img = 255 * np.ones((512, 512, 3), np.uint8)
    testMask = maskCreate(img)
    testMask.drawCanvas()
    
    # Loop until 'q' key is pressed
    while True:
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    del testMask
