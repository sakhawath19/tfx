# Lint as: python2, python3
# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rewriter that invokes the TFLite converter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import logging

import six
import tensorflow.compat.v1 as tf

from tfx.components.trainer.rewriting import rewriter

EXTRA_ASSETS_DIRECTORY = 'assets.extra'


def _copy_dir(src, dst):
  """Recursively copies the contents of the source directory to the destination.

  Args:
    src: Path of the source directory.
    dst: Path of the destination directory.
  """
  for src_dir_name, src_sub_dirs, src_leaf_files in tf.io.gfile.walk(src):
    dst_dir_name = os.path.join(dst, os.path.relpath(src_dir_name, src))
    for leaf_file in src_leaf_files:
      leaf_file_src_path = os.path.join(src_dir_name, leaf_file)
      leaf_file_dst_path = os.path.join(dst_dir_name, leaf_file)
      tf.io.gfile.copy(leaf_file_src_path, leaf_file_dst_path, overwrite=True)

    for sub_dir in src_sub_dirs:
      tf.io.gfile.mkdir(os.path.join(dst_dir_name, sub_dir))


def _create_tflite_converter(saved_model_path,
                             enable_experimental_new_converter):
  converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
  converter.experimental_new_converter = (enable_experimental_new_converter)
  return converter


def _create_tflite_compatible_saved_model(src, dst):
  _copy_dir(src, dst)
  assets_path = os.path.join(dst, tf.saved_model.ASSETS_DIRECTORY)
  if tf.io.gfile.exists(assets_path):
    tf.io.gfile.rmtree(assets_path)
  assets_extra_path = os.path.join(dst, EXTRA_ASSETS_DIRECTORY)
  if tf.io.gfile.exists(assets_extra_path):
    tf.io.gfile.rmtree(assets_extra_path)


class TFLiteRewriter(rewriter.BaseRewriter):
  """Performs TFLite conversion."""

  def __init__(self,
               name,
               filename='tflite',
               enable_experimental_new_converter=False,
               copy_assets=True,
               copy_assets_extra=True):
    """Create an instance of the TFLiteRewriter.

    Args:
      name: The name to use when identifying the rewriter.
      filename: The name of the file to use for the tflite model.
      enable_experimental_new_converter: Whether to use the MLIR converter.
      copy_assets: Boolean whether to copy the assets directory to the rewritten
        model directory.
      copy_assets_extra: Boolean whether to copy the assets.extra directory to
        the rewritten model directory.
    """
    # TODO(dzats): Add additional options for the TFLiteRewriter.
    self._name = name
    self._filename = six.ensure_text(filename)
    self._enable_experimental_new_converter = enable_experimental_new_converter
    self._copy_assets = copy_assets
    self._copy_assets_extra = copy_assets_extra

  @property
  def name(self):
    return self._name

  def pre_rewrite_validate(self, original_model):
    if original_model.model_type != rewriter.ModelType.SAVED_MODEL:
      logging.error('Can only convert SavedModels.')
      return False
    return True

  def rewrite(self, original_model, rewritten_model):
    if rewritten_model.model_type not in [
        rewriter.ModelType.TFLITE_MODEL, rewriter.ModelType.ANY_MODEL
    ]:
      logging.error('Can only convert to the TFLite format.')
      return False

    # TODO(dzats): We create a temporary directory with a SavedModel that does
    # not contain an assets or assets.extra directory. Remove this when the
    # TFLite converter can convert models having these directories.
    tmp_model_dir = os.path.join(six.ensure_text(rewritten_model.path),
                                 'tmp-rewrite-' + str(int(time.time())))
    tf.io.gfile.makedirs(tmp_model_dir)
    _create_tflite_compatible_saved_model(six.ensure_text(original_model.path),
                                          tmp_model_dir)

    converter = _create_tflite_converter(
        tmp_model_dir, self._enable_experimental_new_converter)
    try:
      tflite_model = converter.convert()
    except ValueError as v:
      logging.error(str(v))
      return False
    output_path = os.path.join(
        six.ensure_text(rewritten_model.path), self._filename)
    with tf.io.gfile.GFile(six.ensure_text(output_path), 'wb') as f:
      f.write(six.ensure_binary(tflite_model))
    tf.io.gfile.rmtree(tmp_model_dir)

    copy_pairs = []
    if self._copy_assets:
      src = os.path.join(six.ensure_text(original_model.path),
                         tf.saved_model.ASSETS_DIRECTORY)
      dst = os.path.join(six.ensure_text(rewritten_model.path),
                         tf.saved_model.ASSETS_DIRECTORY)
      if tf.io.gfile.isdir(src):
        tf.io.gfile.mkdir(dst)
        copy_pairs.append((src, dst))
    if self._copy_assets_extra:
      src = os.path.join(six.ensure_text(original_model.path),
                         EXTRA_ASSETS_DIRECTORY)
      dst = os.path.join(six.ensure_text(rewritten_model.path),
                         EXTRA_ASSETS_DIRECTORY)
      if tf.io.gfile.isdir(src):
        tf.io.gfile.mkdir(dst)
        copy_pairs.append((src, dst))
    for src, dst in copy_pairs:
      _copy_dir(src, dst)

    return True

  def post_rewrite_validate(self, rewritten_model):
    # TODO(dzats): Implement post-rewrite validation.
    return True
