import cv2 as cv
class MatchHelper:

    def __init__(self) -> None:

        screen = cv.VideoCapture('E26/screen.avi')
        self.screen_record = []
        while True:
            ret, frame = screen.read()
            if not ret:
                break
            self.screen_record.append(frame)

        hm = cv.VideoCapture('E26/DynamicHeatMap.avi')
        self.heatmap_record = []
        while True:
            ret, frame = hm.read()
            if not ret:
                break
            self.heatmap_record.append(frame)

    def get_similar_images(self, frame_count: int) -> list:
        '''
        Returns a list of images in im_list that are close to im1 by a giving trheshold
        '''
        threshold = 7
        close_images = []
        im1 = cv.cvtColor(self.screen_record[frame_count], cv.COLOR_BGR2GRAY)
        no_match = -float('inf')
        for hm_image in self.heatmap_record[frame_count:]:
            
            hm_image_gray = cv.cvtColor(hm_image, cv.COLOR_BGR2GRAY)
            dif = cv.absdiff(im1, hm_image_gray)

            if dif.mean() < threshold:
              close_images.append(hm_image)
              no_match = 0
            elif no_match > 40:
                break
            else:
                no_match += 1

        
        return close_images