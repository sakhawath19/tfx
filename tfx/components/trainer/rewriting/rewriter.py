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
"""Base class that TFX rewriters inherit and invocation utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import enum

from absl import logging
import six

ModelDescription = collections.namedtuple('ModelDescription',
                                          ['model_type', 'path'])


class ModelType(enum.Enum):
  """Types of models used or created by the rewriter."""
  ANY_MODEL = 1
  SAVED_MODEL = 2
  TFLITE_MODEL = 3


class BaseRewriter(six.with_metaclass(abc.ABCMeta, object)):
  """Base class from which all rewriters should inherit."""

  @abc.abstractproperty
  def name(self):
    """Name of the rewriter.

    Should not be `None` nor empty.
    """
    pass

  @abc.abstractmethod
  def pre_rewrite_validate(self, original_model):
    """Perform pre-rewrite validation to check the model has expected structure.

    Args:
      original_model: A `ModelDescription` object describing the original model.

    Returns:
      Boolean indicating whether the original model has the expected structure.
    """
    pass

  @abc.abstractmethod
  def rewrite(self, original_model, rewritten_model):
    """Perform the rewrite.

    Args:
      original_model: A `ModelDescription` object describing the original model.
      rewritten_model: A `ModelDescription` object describing the location
        and type of the rewritten output.

    Returns:
      Boolean indicating whether the model was successfully rewritten.
    """
    pass

  @abc.abstractmethod
  def post_rewrite_validate(self, rewritten_model):
    """Perform post-rewrite validation.

    Args:
      rewritten_model: A `ModelDescription` object describing the location
        and type of the rewritten output.

    Returns:
      Boolean indicating whether the rewritten model is valid.
    """
    pass


def perform_rewrite(original_model, rewritten_model, rewriter):
  """Invoke all validations and perform the rewrite.

  Args:
    original_model: A `base_rewriter.ModelDescription` object describing the
      original model.
    rewritten_model: A `base_rewriter.ModelDescription` object describing the
      location and type of the rewritten model.
    rewriter: A rewriter instance, which must be a subclass of
      `base_rewriter.BaseRewriter`.

  Returns:
    Boolean indicating whether the rewrite succeeded.
  """
  if not rewriter.pre_rewrite_validate(original_model):
    error_msg = (
        '%s failed to perform pre-rewrite validation. Original model: %s' %
        (rewriter.name, str(original_model)))
    logging.error(error_msg)
    return False

  if not rewriter.rewrite(original_model, rewritten_model):
    error_msg = ('%s failed to rewrite model. Original model: %s' %
                 (rewriter.name, str(original_model)))
    logging.error(error_msg)
    return False

  if not rewriter.post_rewrite_validate(rewritten_model):
    error_msg = ('%s failed to validate rewritten model. Rewritten model: %s'
                 % (rewriter.name, str(rewritten_model)))
    logging.error(error_msg)
    return False

  return True
