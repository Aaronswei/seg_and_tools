#-*-coding:utf-8-*-

import tensorflow as tf

flags = tf.app.flags

# Flags for input preprocessing.

flags.DEFINE_integer('min_resize_value', None,'Desired size of the smaller image side.')

flags.DEFINE_integer('max_resize_value', None,'Maximum allowed size of the larger image side.')

flags.DEFINE_integer('resize_factor', None,'Resized dimensions are multiple of factor plus one.')

# Model dependent flags.
flags.DEFINE_integer('logits_kernel_size', 1,'The kernel size for the convolutional kernel that generates logits.')


# When using 'mobilent_v2', we set atrous_rates = decoder_output_stride = None.
# When using 'xception_65' or 'resnet_v1' model variants, we set
# atrous_rates = [6, 12, 18] (output stride 16) and decoder_output_stride = 4.
# See core/feature_extractor.py for supported model variants.
flags.DEFINE_string('model_variant', 'mobilenet_v2', 'DeepLab model variant.')

flags.DEFINE_multi_float('image_pyramid', None, 'Input scales for multi-scale feature extraction.')

flags.DEFINE_boolean('add_image_level_feature', True,'Add image level feature.')

flags.DEFINE_list(
    'image_pooling_crop_size', None,
    'Image pooling crop size [height, width] used in the ASPP module. When '
    'value is None, the model performs image pooling with "crop_size". This'
    'flag is useful when one likes to use different image pooling sizes.')

flags.DEFINE_list('image_pooling_stride', '1,1','Image pooling stride [height, width] used in the ASPP image pooling. ')

flags.DEFINE_boolean('aspp_with_batch_norm', True,'Use batch norm parameters for ASPP or not.')

flags.DEFINE_boolean('aspp_with_separable_conv', True,'Use separable convolution for ASPP or not.')

# Defaults to None. Set multi_grid = [1, 2, 4] when using provided
# 'resnet_v1_{50,101}_beta' checkpoints.
flags.DEFINE_multi_integer('multi_grid', None,'Employ a hierarchy of atrous rates for ResNet.')

flags.DEFINE_float('depth_multiplier', 1.0,'Multiplier for the depth (number of channels) for all convolution ops used in MobileNet.')

flags.DEFINE_integer('divisible_by', None,'An integer that ensures the layer # channels are divisible by this value. Used in MobileNet.')

# For `xception_65`, use decoder_output_stride = 4. For `mobilenet_v2`, use
# decoder_output_stride = None.
flags.DEFINE_list('decoder_output_stride', [4],
                  'Comma-separated list of strings with the number specifying '
                  'output stride of low-level features at each network level.'
                  'Current semantic segmentation implementation assumes at '
                  'most one output stride (i.e., either None or a list with '
                  'only one element.')

flags.DEFINE_boolean('decoder_use_separable_conv', True,'Employ separable convolution for decoder or not.')

flags.DEFINE_enum('merge_method', 'max', ['max', 'avg'],'Scheme to merge multi scale features.')

flags.DEFINE_boolean(
    'prediction_with_upsampled_logits', True,
    'When performing prediction, there are two options: (1) bilinear '
    'upsampling the logits followed by argmax, or (2) armax followed by '
    'nearest upsampling the predicted labels. The second option may introduce '
    'some "blocking effect", but it is more computationally efficient. '
    'Currently, prediction_with_upsampled_logits=False is only supported for '
    'single-scale inference.')

flags.DEFINE_string('dense_prediction_cell_json','','A JSON file that specifies the dense prediction cell.')

flags.DEFINE_integer('nas_stem_output_num_conv_filters', 20,'Number of filters of the stem output tensor in NAS models.')

flags.DEFINE_bool('use_bounded_activation', False,'Whether or not to use bounded activations. Bounded activations better lend themselves to quantized inference.')

flags.DEFINE_enum(
    'classification_loss', 'softmax_with_attention',
    ['softmax', 'triplet', 'softmax_with_attention'],
    'Type of loss function used for classifying pixels, can be either softmax, '
    'softmax_with_attention, or triplet.')

flags.DEFINE_integer('k_nearest_neighbors', 1,'The number of nearest neighbors to use.')

flags.DEFINE_integer('embedding_dimension', 100, 'The dimension used for the learned embedding')

flags.DEFINE_boolean('use_softmax_feedback', True,'Whether to give the softmax predictions of the last frame as additional input to the segmentation head.')

flags.DEFINE_boolean('sample_adjacent_and_consistent_query_frames', True,
                     'If true, the query frames (all but the first frame '
                     'which is the reference frame) will be sampled such '
                     'that they are adjacent video frames and have the same '
                     'crop coordinates and flip augmentation. Note that if '
                     'use_softmax_feedback is True, this option will '
                     'automatically be activated.')

flags.DEFINE_integer('embedding_seg_feature_dimension', 256,'The dimensionality used in the segmentation head layers.')

flags.DEFINE_integer('embedding_seg_n_layers', 4, 'The number of layers in the segmentation head.')

flags.DEFINE_integer('embedding_seg_kernel_size', 7, 'The kernel size used in the segmentation head.')

flags.DEFINE_multi_integer('embedding_seg_atrous_rates', [],'The atrous rates to use for the segmentation head.')

flags.DEFINE_boolean('normalize_nearest_neighbor_distances', True,'Whether to normalize the nearest neighbor distances to [0,1] using sigmoid, scale and shift.')

flags.DEFINE_boolean('also_attend_to_previous_frame', True, 'Whether to also use nearest neighbor attention with respect to the previous frame.')

flags.DEFINE_bool('use_local_previous_frame_attention', True,
                  'Whether to restrict the previous frame attention to a local '
                  'search window. Only has an effect, if '
                  'also_attend_to_previous_frame is True.')

flags.DEFINE_integer('previous_frame_attention_window_size', 15,'The window size used for local previous frame attention, if use_local_previous_frame_attention is True.')

flags.DEFINE_boolean('use_first_frame_matching', True, 'Whether to extract features by matching to the reference frame. This should always be true except for ablation experiments.')


FLAGS = flags.FLAGS