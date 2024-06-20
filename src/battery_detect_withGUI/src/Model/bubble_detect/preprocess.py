import numpy as np
import cv2 as cv
from enum import Enum
from matplotlib import pyplot as plt
import csv

class bubble:
    def __init__(self):
        self.contour = []
        self.area = []
        self.boundingBoxes = []

    def setInputContourAndArea(self, contour, area):
        self.contour = contour
        self.area = area


    def sort(self, contourAreaThres):
        sortArea = self.area[self.area >= 0]
        sortContour = [self.contour[i] for i in range(len(self.contour)) if self.area[i] >= contourAreaThres]

        self.area = sortArea
        self.contour = sortContour

    def calcBubbleCoverRatio(self, img_size):
        return np.sum(self.area)/img_size*100
    
    def saveBubbleFile(self):
        header = ['x', 'y', 'width', 'height', 'centerX', 'centerY']
        # Set the output filename 
        output_dir = f'../data/bubble_info/'
        outputFilename = output_dir + "bubble_box_data.csv"
        with open(outputFilename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(self.boundingBoxes)

class blurMode(Enum):
    average = 0
    gaussian = 1
    median= 2
    bilateral = 3

class threshMode(Enum):
    simple = 0
    adaptive = 1
    otsu = 2

class preprocess:
    def __init__(self):
        print("Create preprocess class") 
        self.Bubble = bubble()
        self.inputImg = []
        self.imgWaitToHist = []
        self.tmpImg = []
        self.threshValue = []
        self.procImg = []                           # Processed image
        self.erodeKernel_0 = self.createKernel(size=6)   # Create Erosion Kernel    
        self.dilateKernel_0 =self.createKernel(size=6)   # Create Dilation Kernel
    
    def setInputImg(self, inputImg):
        # Check the canvas is grayscale or not.
        # If No, the inputImg should be converted to Gray scale
        if self.checkImgIsGray(inputImg) == 0:
            self.inputImg = inputImg
        else:
            self.inputImg = cv.cvtColor(inputImg, cv.COLOR_BGR2GRAY)
    
    # Create a kernel consisting of ones
    def createKernel(self, size):
        return np.ones((size,size), np.uint8)
    
    def blur(self, inputImg, kernelSize, mode):
        if mode == 0:
            print("Averaging Blurring")
            return cv.blur(inputImg, (kernelSize,kernelSize))
        elif mode == 1:
            print("Gaussian Blurring")
            return cv.GaussianBlur(inputImg, (kernelSize,kernelSize), 0)
        elif mode == 2:
            print("Median Blurring")
            return cv.medianBlur(inputImg, kernelSize)
        elif mode == 3:
            print("Bilateral Filtering")
            return cv.bilateralFilter(inputImg, 9, 75, 75)
        else:
            print("Invalid code for image blurring")

    def erode(self, inputImg, kernel, iteration):
        return cv.erode(inputImg, kernel, iteration)
    
    def dilate(self, inputImg, kernel, iteration):
        return cv.dilate(inputImg, kernel, iteration)
    
    def threshold(self, inputImg, thresh, maxval, type, mode, adaptiveMethod=cv.ADAPTIVE_THRESH_MEAN_C):
        if mode == 0:
            print("Simple thresholding: thresh ={}".format(thresh))
            ret, th = cv.threshold(inputImg, thresh, maxval, type)
            self.threshValue = ret
            return th
        elif mode == 1:
            # https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3
            print("Adaptive thresholding")
            th = cv.adaptiveThreshold(inputImg, maxval, adaptiveMethod, type, blockSize=1081, C=0)
            return th
        elif mode == 2:
            print("Otsu's thresholding")
            ret, th = cv.threshold(inputImg, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
            self.threshValue = ret
            return th
        else:
            print("Invalid code for thresholding")

    def preprocess(self):
        self.tmpImg = self.inputImg
        # cv.imshow("Input image", cv.resize(self.tmpImg, (640, 480)))
        # cv.waitKey(0)

        # Blur
        self.tmpImg = self.blur(self.inputImg, kernelSize = 9, mode=blurMode.median.value)
        # cv.imshow("test 1: Blurring", cv.resize(self.tmpImg, (640, 480)))
        # cv.waitKey(0)

        # Enhancement
        # self.tmpImg = cv.equalizeHist(self.tmpImg)
        # cv.imshow("test ?: Enhancement", self.tmpImg)
        # cv.waitKey(0)

        # Backup the image to show its histogram
        self.imgWaitToHist = self.tmpImg.copy()

        # Threshold
        selfDefinedTh = self.twoPeaksLowest(self.tmpImg)
        self.tmpImg = self.threshold(self.tmpImg, selfDefinedTh, 255, type=cv.THRESH_BINARY, mode=threshMode.simple.value)
        print("Threshold value ={}".format(self.threshValue))
        # cv.imshow("test 2: Thresholding", self.tmpImg)
        # cv.waitKey(0)

        # Dilation
        self.tmpImg = self.dilate(self.tmpImg, self.dilateKernel_0, iteration=1)
        # cv.imshow("test 3: Dilation", self.tmpImg)
        # cv.waitKey(0)

        # Erosion
        self.tmpImg = self.erode(self.tmpImg, self.erodeKernel_0, iteration=1)
        # cv.imshow("test 4: Erosion", self.tmpImg)
        # cv.waitKey(0)

        # Erosion
        self.tmpImg = self.erode(self.tmpImg, self.erodeKernel_0, iteration=1)
        # cv.imshow("test 5: Erosion", self.tmpImg)
        # cv.waitKey(0)

        # Dilation
        self.tmpImg = self.dilate(self.tmpImg, self.dilateKernel_0, iteration=1)
        # cv.imshow("test 6: Dilation", self.tmpImg)
        # cv.waitKey(0)

        # cv.destroyAllWindows()

        # Output
        self.procImg = self.tmpImg
        print("Preprocess image shape = {}".format(self.procImg.shape))

        return self.procImg

    # Warning: If there were too many noise exist on the image, contour would find just one contour(I guess)
    def findBubble(self):
        canvas = self.procImg.copy()
        boundingBoxes = []

        # Check the canvas is grayscale or not.
        # If yes, the canvas should be converted to BGR color scale
        if self.checkImgIsGray(canvas) == 0:
            canvas = cv.cvtColor(canvas, cv.COLOR_GRAY2BGR)

        # Finding Contours
        contours, _ = cv.findContours(self.procImg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Create Bubble class to record bubble's information
        self.Bubble.setInputContourAndArea(contour = contours, area = [])

        # Check contours was found or not
        if len(contours) != 0:
            # Calculate the area of the contour
            # for contour in contours:
            #     Bubble.area.append(cv.contourArea(contour))  
            # Bubble.area = np.array(Bubble.area)
            
            # Same as above
            self.Bubble.area = np.array([cv.contourArea(contour) for contour in contours])

            # Sort bubble with minimum area size
            self.Bubble.sort(contourAreaThres=0)

            # Draw Bubble's information
            for i, contour in enumerate(self.Bubble.contour):
                # Draw the bonding box
                x, y, w, h = cv.boundingRect(contour)
                cv.rectangle(canvas, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Append bounding box to array
                self.Bubble.boundingBoxes.append([x, y, w, h, x+w/2, y+h/2])

                # Create bonding box label
                label = f'size={self.Bubble.area[i]:.0f}'
                cv.putText(canvas, label, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv.circle(canvas, (int(x+w/2), int(y+h/2)), radius=5, color=(0, 0, 255), thickness=-1)
            
            # Display the image
            # cv.imshow('Bubble label', cv.resize(canvas, (640, 480)))
            # cv.imshow('Bubble label', canvas)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
        else:
            print("Doesn't find bubble (No contours)")

        return canvas, self.Bubble
    
    # Check the image is gray scale or not
    def checkImgIsGray(self, img):
        if len(img.shape) == 2:
            print("Grayscale image")
            return 0
        elif len(img.shape) == 3 and img.shape[2] == 3:
            print("Color Scale image")
            return 1
        else:
            print("Not a grayscale or color scale image")
            return -1
    
    # Find the lowest point between two peaks as threshold
    def twoPeaksLowest(self, inputImg):
        # Get the input image's histogram
        histogram = cv.calcHist([inputImg], [0], None, [256], [0, 256])

        # Reshape Histogram from 2d array (256, 1) to 1d array (256, )
        histogram = histogram.T[0, :]

        # Create x and y data to draw histogram
        plot_y_range = np.array(range(256))
        plot_x_range = histogram

        # Define the sliding window size
        window_size = 5

        # Perform Moving Average (with Convolution)
        filtered_signal = self.movingAverage(histogram, window_size)

        # Find the maximum and minimum of the histogram curve
        hist1d = filtered_signal
        expand_hist1d = np.insert(hist1d, 0, hist1d[0])
        expand_hist1d = np.insert(expand_hist1d, len(hist1d), hist1d[-1])

        # Calculate the difference between the current element and the previous element
        diff_hist1d = np.diff(expand_hist1d)

        # Find the turning point and classify it is lowest or highest.
        # If the previous value is less than 0 and subsequent value is greater than 0, the turning points should be the minimum value
        # If the previous value is greater than 0 and subsequent value is less than 0, the turning points should be the maximum value
        # If both the previous value and subsequent value are equal to zero, there is no turning point
        mask_min = (diff_hist1d[:-1] < 0) & (diff_hist1d[1:] > 0)
        mask_max = (diff_hist1d[:-1] > 0) & (diff_hist1d[1:] < 0)
        turningPt = np.zeros(len(diff_hist1d)-1, dtype="uint8")
        turningPt[mask_min] = 1
        turningPt[mask_max] = 2
        
        # print("Maximum in histogram is{}".format(np.where(turningPt == 2)))
        # print("Minimum in histogram is{}".format(np.where(turningPt == 1)))
        # print("Length of filter signal={}".format(len(filtered_signal)))

        # ########## Plot ##########
        # plt.bar(plot_y_range, plot_x_range)
        # plt.plot(plot_x_range, color='blue', label='original histogram')
        # plt.plot(filtered_signal, color='red', label='smoothing histogram')
        # plt.legend()
        # plt.show()

        # return threshold (first minimum)
        print(np.where(turningPt == 1))
        return np.where(turningPt == 1)[0].astype(int)[0]

    def movingAverage(self, signal1d, windowSize):
        # Create sliding window
        slidingWindow = np.ones(windowSize) / windowSize

        # padSize = int((slidingWindow.size -1) / 2)
        # signal1dPadded = np.pad(signal1d, (padSize, padSize), mode="constant",constant_values=(0))
        
        # # Compute the size of the output array
        # output_size = signal1d.size
        
        # # Create an empty array to hold the output
        # output = np.zeros(output_size)
        
        # # Perform convolution using a loop
        # for i in range(output_size):
        #     output[i] = np.sum(signal1dPadded[i:i+windowSize] * slidingWindow)

        # Same as above
        output = np.convolve(signal1d, slidingWindow, mode='same')
        return output