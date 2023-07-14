from heatmap_extractor import HeatmapCenterExtractor
from contour_avg_ploter import PlotContourAvg

import cv2 as cv
import numpy as np


class SalientValidator:

    def __init__(self, screen_record: list, heatmap_record: list, salience: list) -> None:
        '''
        parameters:
            screen_record (list): list of path for each frame of the screen recording
            heatmap_record (list): list of path for each frame of the heatmap recording
        '''
        #print('Loading frames...')
        self.screen_record = [cv.imread(img) for img in sorted(screen_record, key= lambda x: int(x.split('/')[-1].split('.')[0]))]
        self.heatmap_record = [cv.imread(img) for img in sorted(heatmap_record, key= lambda x: int(x.split('/')[-1].split('.')[0]))]
        self.salience_record = [cv.imread(img, cv.IMREAD_GRAYSCALE) for img in sorted(salience, key= lambda x: int(x.split('/')[-1].split('.')[0]))]
        #print('Frames loaded')
        self.experiment = screen_record[0].split('/')[-3]
        self.heatmap_extractor = HeatmapCenterExtractor()
        self.countour_ploter = PlotContourAvg()

    def __get_brighter_pixel(self, frame: np.array) -> tuple:
        '''
        Returns a list with all brighter pixels in the frame

        Parameters:
            frame (np.array): binary image to be processed

        Returns:
            list[tuple]: A list with all brighter pixels in the frame
        '''
        brighter_pixel = np.unravel_index(np.argmax(frame), frame.shape)
        # get all pixels with the same value as the brighter pixel
        values = np.where(frame == frame[brighter_pixel])
        return list(zip(values[0], values[1]))

    def __get_contours(self, frame: np.array) -> list:
        '''
        Returns a list of contours in the frame

        Parameters:
            frame (np.array): binary image to be processed

        Returns:
            list: A list of contours in the frame
        '''
        _, bi = cv.threshold(frame, 50, 255, 0)
        contours, _ = cv.findContours(bi, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        return contours
    
    def __get_contour_center(self, contour):
        # for each contour get the center
        centers = []
        for c in contour:
            M = cv.moments(c)
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY))
        return centers
        
    def __get_heatmap_center(self, screen: np.array, heatmap: np.array) -> tuple:

        dif = cv.absdiff(cv.cvtColor(screen, cv.COLOR_BGR2GRAY),
                        cv.cvtColor(heatmap, cv.COLOR_BGR2GRAY))

        _, bi = cv.threshold(dif,15,255,0)

        if det := self.heatmap_extractor.get_salience_center(bi):
            return det
        return None
    
    def __get_heatmap_mask(self, screen: np.array, heatmap: np.array):

        diff = cv.absdiff(cv.cvtColor(screen, cv.COLOR_BGR2GRAY),
                        cv.cvtColor(heatmap, cv.COLOR_BGR2GRAY))
        
        _, bi = cv.threshold(diff,15,255,0)

        contour = self.heatmap_extractor.get_salience_contour(bi)
        if contour is not None:
            contour = contour.contour
        else:
            return None
        #create a mask of 0 with the area indise the contour as 1
        mask = np.zeros(screen.shape[:2], dtype=np.uint8)
        cv.drawContours(mask, [contour], -1, color=(255, 255, 255), thickness=cv.FILLED)
        return mask

    def __get_pixels_with_value(self, image: np.array, value: int) -> int:
        return np.count_nonzero(image == value)
    
    def __get_area_inside_heatmap(self, image: np.array, mask: np.array) -> np.array:

        # filter the salience area inside the heatmap mask
        salience = cv.bitwise_and(image, image, mask=mask)

        return salience
    
    def __get_gaze_sal_comparison(self, screen: np.array, salience: np.array, heatmap: np.array) -> float:
        '''
        Get the percentage of salience area inside the heatmap

        Parameters:
            screen (np.array): screen frame
            salience (np.array): salience frame
            heatmap (np.array): heatmap frame

        Returns:
            float: avrage salience pixel brightness inside the heatmap detection area
        '''
        heatmap_mask = self.__get_heatmap_mask(screen, heatmap)

        if heatmap_mask is None:
            return None
        
        mask_size = self.__get_pixels_with_value(heatmap_mask, 255)

        sal_area = self.__get_area_inside_heatmap(salience, heatmap_mask)
        
        # sum all pixels inside the sal area
        sal_sum = np.sum(sal_area)

        return sal_sum / mask_size

    def __merge_frames(self, screen: np.ndarray, salience: np.ndarray) -> np.ndarray:
        '''
        Merge screen and salience frames

        '''
        salience = cv.cvtColor(salience, cv.COLOR_GRAY2BGR)
        return cv.addWeighted(screen, 0.3, salience, 0.6, 0)
            
    def __is_eye_close(self, screen_f: np.array, frame_count: int, attention_points: list):
        window_frames = 10

        start = max(0, frame_count - window_frames//2)
        end = min(len(self.heatmap_record), frame_count + window_frames//2+1)
        
        height, width, _ = screen_f.shape

        dets = [self.__get_heatmap_center(screen_f, h_frame) for h_frame in self.heatmap_record[start:end]]
        if all(det is None for det in dets):
            return -1, None
        min_dist = float('inf')
        close_point = None
        for point in attention_points:
            for det in dets:
                if det is None:
                    continue
                dist = abs(complex(*det) - complex(*point))
                if dist < min_dist:
                    min_dist = dist
                    close_point = det
        return min_dist, close_point
    
    def validate_salience(self):
        
        dists = []
        height, width, _ = self.screen_record[0].shape
        center = (width//2, height//2)

        for frame_count, frames in enumerate(zip(self.screen_record, self.salience_record)):
            screen_f, sal_f = frames
            heatmap_f = self.heatmap_record[frame_count]

            pixels_avg_brightness = self.__get_gaze_sal_comparison(screen_f, sal_f, heatmap_f)
            if pixels_avg_brightness is not None:
                dists.append(self.__get_gaze_sal_comparison(screen_f, sal_f, heatmap_f))
            #attention_points = self.__get_brighter_pixel(sal_f) #self.__get_contour_center(self.__get_contours(sal_f))#
            
            
            # hm_center = self.__get_heatmap_center(screen_f, self.heatmap_record[frame_count])
            # if hm_center is not None:
            #     dist = abs(complex(*center) - complex(*hm_center))
            #     dists.append(dist)
            # dist, _ = self.__is_eye_close(screen_f, frame_count, attention_points)
            # if dist > 0:
            #     dists.append(dist)


        return dists

            
    def create_visualization(self):

        height, width, _ = self.screen_record[0].shape
        video = cv.VideoWriter(f'videos/{self.experiment}.mp4', cv.VideoWriter_fourcc(*'mp4v'), 5, (width,height))

        for frame_count, frame in enumerate(zip(self.screen_record, self.salience_record)):
            
            screen_f, sal_f = frame
            video_frame = self.__merge_frames(screen_f, sal_f)
            
            #attention_points = self.__get_contour_center(self.__get_contours(sal_f))
            attention_points = self.__get_brighter_pixel(sal_f)

            _, det = self.__is_eye_close(screen_f, frame_count, attention_points)

            if det is not None:
                cv.circle(video_frame, det[::-1], 10, (0,0,255), -1)

            for point in attention_points:
                cv.circle(video_frame, point[::-1], 10, (255,0,0), -1)

            video.write(video_frame)

        video.release()

    def create_sal_visualization(self):

        height, width, _ = self.screen_record[0].shape
        video = cv.VideoWriter(f'videos/{self.experiment}.mp4', cv.VideoWriter_fourcc(*'mp4v'), 5, (width,height))

        for frame_count, frame in enumerate(zip(self.screen_record, self.salience_record)):
            
            screen_f, sal_f = frame

            try:

                video_frame = self.__merge_frames(screen_f, sal_f)
            except:
                continue

            
            #attention_points = self.__get_contour_center(self.__get_contours(sal_f))

            #attention_points = self.__get_brighter_pixel(sal_f)

            #for point in attention_points:
            #    cv.circle(video_frame, point[::-1], 5, (255,0,0), -1)

            #self.countour_ploter.draw_avg_values(sal_f, video_frame)

            video.write(video_frame)

        video.release()