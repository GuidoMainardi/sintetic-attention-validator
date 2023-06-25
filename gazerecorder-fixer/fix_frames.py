import pygame
import cv2 as cv
import numpy as np
from get_similar_frames import MatchHelper
import os

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

def start():

    pygame.init()

    frame = (SCREEN_WIDTH, SCREEN_HEIGHT)
    return pygame.display.set_mode(frame)
    


def tick(screen):
    
    mh = MatchHelper()

    curr_s_frame = 0
    hm_frame = 0
    similar_frames = mh.get_similar_images(curr_s_frame)
    while True:
        
        curr_screen_frame = mh.screen_record[curr_s_frame]
        try:
            curr_heatmap_frame = similar_frames[hm_frame]
        except IndexError:
            curr_s_frame += 1
            curr_screen_frame = mh.screen_record[curr_s_frame]
            similar_frames = mh.get_similar_images(curr_s_frame)
            hm_frame = 0
            continue
            
        display_frames(screen, curr_screen_frame, curr_heatmap_frame)
        
        key = pygame.key.get_pressed()
        if key[pygame.K_ESCAPE]:
            break
            
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    try:
                        hm_frame = (hm_frame + 1) % len(similar_frames)
                    except ZeroDivisionError:
                        pass
                if event.key == pygame.K_LEFT:
                    try:
                        hm_frame = (hm_frame - 1) % len(similar_frames)
                    except ZeroDivisionError:
                        pass
                if event.key == pygame.K_DOWN:
                    curr_s_frame -= 1
                    try:
                        curr_screen_frame = mh.screen_record[curr_s_frame]
                    except IndexError:
                        break
                    similar_frames = mh.get_similar_images(curr_s_frame)
                    hm_frame = 0
                if event.key == pygame.K_UP:
                    curr_s_frame += 1
                    try:
                        curr_screen_frame = mh.screen_record[curr_s_frame]
                    except IndexError:
                        break
                    similar_frames = mh.get_similar_images(curr_s_frame)
                    hm_frame = 0
                if event.key == pygame.K_s:
                    cv.imwrite(f'outE26/screen/{str(curr_s_frame).zfill(4)}.png', 
                               curr_screen_frame)
                    cv.imwrite(f'outE26/heatmap/{str(curr_s_frame).zfill(4)}.png', 
                               curr_heatmap_frame)
                    screen.fill((100, 100, 100))
                    curr_s_frame += 1
                    try:
                        curr_screen_frame = mh.screen_record[curr_s_frame]
                    except IndexError:
                        break
                    similar_frames = mh.get_similar_images(curr_s_frame)
                    hm_frame = 0
                if event.key == pygame.K_n:
                    curr_s_frame += 1
                    try:
                        curr_screen_frame = mh.screen_record[curr_s_frame]
                    except IndexError:
                        break
                    similar_frames = mh.get_similar_images(curr_s_frame)
                    hm_frame = 0
                if event.key == pygame.K_q:
                    # quit the program
                    break
        pygame.display.update()

    pygame.quit()
def display_frames(screen, f1, f2):
    # rotate frames 90 degrees
    f1 = cv.rotate(f1, cv.ROTATE_90_COUNTERCLOCKWISE)
    f2 = cv.rotate(f2, cv.ROTATE_90_COUNTERCLOCKWISE)    

    # convert to grayscale
    f1 = cv.cvtColor(f1, cv.COLOR_BGR2GRAY)
    f2 = cv.cvtColor(f2, cv.COLOR_BGR2GRAY)

    # abs diff between frames
    diff = cv.absdiff(f1, f2)

    diff = cv.resize(diff, (720, 1280))
    # convert frames to surface
    diff = pygame.surfarray.make_surface(np.stack((diff,)*3, axis=-1))
    #f2 = pygame.surfarray.make_surface(f2)

    # display both frames
    screen.blit(diff, (0,0))
    #screen.blit(f2, (600, 0))

if __name__ == '__main__':

    if not os.path.exists('outE26'):
        os.mkdir('outE26')
    if not os.path.exists('outE26/screen'):
        os.mkdir('outE26/screen')
    if not os.path.exists('outE26/heatmap'):
        os.mkdir('outE26/heatmap')

    screen= start()

    tick(screen)

        