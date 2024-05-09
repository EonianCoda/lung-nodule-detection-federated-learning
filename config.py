SAVE_ROOT = './save'
DEFAULT_OVERLAP_RATIO = 0.25
DEFAULT_CROP_SIZE = [96, 96, 96]
IMAGE_SPACING = [1.0, 0.8, 0.8]

NODULE_TYPE_DIAMETERS = {'benign': [0,4],
                        'probably_benign': [4, 6],
                        'probably_suspicious': [6, 8],
                        'suspicious': [8, -1]}