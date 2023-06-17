import os
from salient_validator import SalientValidator

screen_frames = [f'E2/screen/{frame}' for frame in os.listdir('E2/screen')]
hm_frames = [f'E2/heatmap/{frame}' for frame in os.listdir('E2/heatmap')]
sal_frames = [f'E2/sal/{frame}' for frame in os.listdir('E2/sal')]

sv = SalientValidator(screen_frames, hm_frames, sal_frames)

print(f'model accuracy {sv.validate_salience()}')

sv.create_visualization()