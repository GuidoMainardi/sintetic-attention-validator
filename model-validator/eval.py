import os
from salient_validator import SalientValidator
import math


def mrse(dists: list):
    return math.sqrt(sum(map(lambda x: x**2, dists)) / len(dists))

def load_videos(path: str) -> tuple:

    screen_frames = [f'{path}/screen/{frame}' 
                        for frame in os.listdir(f'{path}/screen') 
                            if frame != '.DS_Store']
    
    hm_frames = [f'{path}/heatmap/{frame}' 
                    for frame in os.listdir(f'{path}/heatmap') 
                        if frame != '.DS_Store']

    sal_frames = [f'{path}/sal/{frame}' 
                    for frame in os.listdir(f'{path}/sal') 
                        if frame != '.DS_Store']

    return screen_frames, hm_frames, sal_frames

total_dists = []

# for case in data file dir append the results to total_dists
for case in os.listdir('data/demo'):
    if case == '.DS_Store':
        continue

    print(f' validating {case}')
    screen_frames, hm_frames, sal_frames = load_videos(f'data/demo/{case}')
    sv = SalientValidator(screen_frames, hm_frames, sal_frames)
    dists = sv.validate_salience()
    total_dists.extend(dists)
    print(f'Parcial MRSE: {mrse(total_dists)}')

print(f'Total MRSE: {mrse(total_dists)}')