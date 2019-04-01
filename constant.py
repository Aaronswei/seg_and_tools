#-*-coding:utf-8-*-

WRONG_LABEL_PADDING_DISTANCE = 1e20
MIN_LABEL_COUNT = 10

# The probability of flipping the images and labels
# left-right during training
_PROB_OF_FLIP = 0.5

# Local constants.
_META_ARCHITECTURE_SCOPE = 'meta_architecture'
_CONCAT_PROJECTION_SCOPE = 'concat_projection'
_OP = 'op'
_CONV = 'conv'
_PYRAMID_POOLING = 'pyramid_pooling'
_KERNEL = 'kernel'
_RATE = 'rate'
_GRID_SIZE = 'grid_size'
_TARGET_SIZE = 'target_size'
_INPUT = 'input'


# Constants

# Perform semantic segmentation predictions.
OUTPUT_TYPE = 'semantic'

# Semantic segmentation item names.
LABELS_CLASS = 'labels_class'
IMAGE = 'image'
HEIGHT = 'height'
WIDTH = 'width'
IMAGE_NAME = 'image_name'
SOURCE_ID = 'source_id'
VIDEO_ID = 'video_id'
LABEL = 'label'
ORIGINAL_IMAGE = 'original_image'
PRECEDING_FRAME_LABEL = 'preceding_frame_label'

# Test set name.
TEST_SET = 'test'

# Internal constants.
OBJECT_LABEL = 'object_label'


