import os
from salient_validator import SalientValidator
import math


def mrse(dists: list):
    return math.sqrt(sum(map(lambda x: x**2, dists)) / len(dists))

def mae(dists: list):
    return sum(dists) / len(dists)

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
path = 'data/Video'
acepted = 0
for video, case in enumerate(os.listdir(path)):
    if case == '.DS_Store':
        continue

    print(f' validating {case}')
    screen_frames, hm_frames, sal_frames = load_videos(f'{path}/{case}')
    sv = SalientValidator(screen_frames, hm_frames, sal_frames)
    dists = sv.validate_salience()
    
    print(f'\tvideo {video} frames count: {len(dists)}')
    print(f'\tavg pixel brighness video {video}: {sum(dists) / len(dists)}')
    print(f'\tsalience miss rate video {video}: {len([d for d in dists if d == 0]) / len(dists)}')
    print(f'\tsucceded detected frame rate {len(dists)/463}')
    if len(dists)/463 > 0.8:
        total_dists.extend(dists)
        acepted += 1
    print()
print(f'cumulative frames count: {len(total_dists)}')
print(f'cumulative avg pixel brighness: {sum(total_dists) / len(total_dists)}')
print(f'cumulative salience miss rate: {len([d for d in total_dists if d == 0]) / len(total_dists)}')
print(f'cumulative succeded detected frame rate {len(total_dists)/(463*acepted)}')
print()
#     print(f'Parcial MRSE: {mrse(total_dists)}')
#     print(f'Parcial MAE: {mae(total_dists)}')

# print(f'Total MRSE: {mrse(total_dists)}')
# print(f'Total MAE: {mae(total_dists)}')