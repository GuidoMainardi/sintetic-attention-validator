import cv2 as cv
import numpy as np 

class PlotContourAvg:

    def __init__(self) -> None:
        pass

    def __get_pixels_inside(self, img, contour):
        mask = np.zeros_like(img)
        cv.drawContours(mask, [contour], -1, (255,255,255), -1)
        return cv.bitwise_and(img, mask)

    def __get_contours(self, img):
        _, bi = cv.threshold(img,35,255,0)
        contours, _ = cv.findContours(bi, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        return contours
    
    def __get_avg_value_area(self, img, contour):
        pixels = self.__get_pixels_inside(img, contour)
        return int(np.mean(pixels[pixels != 0]))
        
    
    def __get_contour_center(self, contour):
        M = cv.moments(contour)
        if M["m00"] == 0:
            return (0, 0)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)
    
    def draw_avg_values(self, base, out):
        for c in self.__get_contours(base):

            center = self.__get_contour_center(c)

            if center == (0, 0):
                continue

            avg = self.__get_avg_value_area(base, c)

            cv.rectangle(out, (center[0]-5, center[1]-50), (center[0]+100, center[1]+5), (0,0,0), -1)

            cv.putText(out, str(avg), center, cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv.LINE_AA)