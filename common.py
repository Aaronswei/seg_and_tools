#-*-coding:utf-8-*-
"""Provides flags that are common to scripts.

Common flags from train/vis_video.py are collected in this script.
"""
import tensorflow as tf
import collections
import copy
import json

from settings import FLAGS
from constant import *


class ModelOptions(
    collections.namedtuple('ModelOptions', [
        'outputs_to_num_classes',
        'crop_size',
        'atrous_rates',
        'output_stride',
        'preprocessed_images_dtype',
        'merge_method',
        'add_image_level_feature',
        'image_pooling_crop_size',
        'image_pooling_stride',
        'aspp_with_batch_norm',
        'aspp_with_separable_conv',
        'multi_grid',
        'decoder_output_stride',
        'decoder_use_separable_conv',
        'logits_kernel_size',
        'model_variant',
        'depth_multiplier',
        'divisible_by',
        'prediction_with_upsampled_logits',
        'dense_prediction_cell_config',
        'nas_stem_output_num_conv_filters',
        'use_bounded_activation'
    ])):
  """Immutable class to hold model options."""

  __slots__ = ()

  def __new__(cls,
              outputs_to_num_classes,
              crop_size=None,
              atrous_rates=None,
              output_stride=8,
              preprocessed_images_dtype=tf.float32):
    """Constructor to set default values.

    Args:
      outputs_to_num_classes: A dictionary from output type to the number of
        classes. For example, for the task of semantic segmentation with 21
        semantic classes, we would have outputs_to_num_classes['semantic'] = 21.
      crop_size: A tuple [crop_height, crop_width].
      atrous_rates: A list of atrous convolution rates for ASPP.
      output_stride: The ratio of input to output spatial resolution.
      preprocessed_images_dtype: The type after the preprocessing function.

    Returns:
      A new ModelOptions instance.
    """
    dense_prediction_cell_config = None
    if FLAGS.dense_prediction_cell_json:
      with tf.gfile.Open(FLAGS.dense_prediction_cell_json, 'r') as f:
        dense_prediction_cell_config = json.load(f)
    decoder_output_stride = None
    if FLAGS.decoder_output_stride:
      decoder_output_stride = [
          int(x) for x in FLAGS.decoder_output_stride]
      if sorted(decoder_output_stride, reverse=True) != decoder_output_stride:
        raise ValueError('Decoder output stride need to be sorted in the '
                         'descending order.')
    image_pooling_crop_size = None
    if FLAGS.image_pooling_crop_size:
      image_pooling_crop_size = [int(x) for x in FLAGS.image_pooling_crop_size]
    image_pooling_stride = [1, 1]
    if FLAGS.image_pooling_stride:
      image_pooling_stride = [int(x) for x in FLAGS.image_pooling_stride]
    return super(ModelOptions, cls).__new__(
        cls, outputs_to_num_classes, crop_size, atrous_rates, output_stride,
        preprocessed_images_dtype, FLAGS.merge_method,
        FLAGS.add_image_level_feature,
        image_pooling_crop_size,
        image_pooling_stride,
        FLAGS.aspp_with_batch_norm,
        FLAGS.aspp_with_separable_conv, FLAGS.multi_grid, decoder_output_stride,
        FLAGS.decoder_use_separable_conv, FLAGS.logits_kernel_size,
        FLAGS.model_variant, FLAGS.depth_multiplier, FLAGS.divisible_by,
        FLAGS.prediction_with_upsampled_logits, dense_prediction_cell_config,
        FLAGS.nas_stem_output_num_conv_filters, FLAGS.use_bounded_activation)

  def __deepcopy__(self, memo):
    return ModelOptions(copy.deepcopy(self.outputs_to_num_classes),
                        self.crop_size,
                        self.atrous_rates,
                        self.output_stride,
                        self.preprocessed_images_dtype)




class VideoModelOptions(ModelOptions):
  """Internal version of immutable class to hold model options."""

  def __new__(cls,
              outputs_to_num_classes,
              crop_size=None,
              atrous_rates=None,
              output_stride=8):
    """Constructor to set default values.

    Args:
      outputs_to_num_classes: A dictionary from output type to the number of
        classes. For example, for the task of semantic segmentation with 21
        semantic classes, we would have outputs_to_num_classes['semantic'] = 21.
      crop_size: A tuple [crop_height, crop_width].
      atrous_rates: A list of atrous convolution rates for ASPP.
      output_stride: The ratio of input to output spatial resolution.

    Returns:
      A new VideoModelOptions instance.
    """
    self = super(VideoModelOptions, cls).__new__(
        cls,
        outputs_to_num_classes,
        crop_size,
        atrous_rates,
        output_stride)
    # Add internal flags.
    self.classification_loss = FLAGS.classification_loss

    return self


def parse_decoder_output_stride():
  """Parses decoder output stride.

  FEELVOS assumes decoder_output_stride = 4. Thus, this function is created for
  this particular purpose.

  Returns:
    An integer specifying the decoder_output_stride.

  Raises:
    ValueError: If decoder_output_stride is None or contains more than one
      element.
  """
  if FLAGS.decoder_output_stride:
    decoder_output_stride = [
        int(x) for x in FLAGS.decoder_output_stride]
    if len(decoder_output_stride) != 1:
      raise ValueError('Expect decoder output stride has only one element.')
    decoder_output_stride = decoder_output_stride[0]
  else:
    raise ValueError('Expect flag decoder output stride not to be None.')
  return decoder_output_stride
