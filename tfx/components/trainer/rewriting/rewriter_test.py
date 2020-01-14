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
"""Tests for third_party.components.trainer.rewriting.rewriter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from tfx.components.trainer.rewriting import rewriter


class PerformRewriteTest(absltest.TestCase):

  class MyRewriter(rewriter.BaseRewriter):

    def __init__(self, pre_rewrite_result, rewrite_result, post_rewrite_result,
                 expected_original_model, expected_rewritten_model):
      """Initializes the MyRewriter class.

      Args:
        pre_rewrite_result: The boolean return for pre rewrite validation
        rewrite_result: The boolean to return for rewrite.
        post_rewrite_result: The boolean to return for post rewrite validation.
        expected_original_model: A `rewriter.ModelDescription` specifying
          the expected original model.
        expected_rewritten_model: A `rewriter.ModelDescription` specifying
          the expected rewritten model.
      """
      self._pre_rewrite_result = pre_rewrite_result
      self._rewrite_result = rewrite_result
      self._post_rewrite_result = post_rewrite_result
      self._expected_original_model = expected_original_model
      self._expected_rewritten_model = expected_rewritten_model

      self.pre_rewrite_validate_called = False
      self.rewrite_called = False
      self.post_rewrite_validate_called = False

    @property
    def name(self):
      return 'my_rewriter'

    def pre_rewrite_validate(self, original_model):
      assert original_model == self._expected_original_model
      self.pre_rewrite_validate_called = True
      return self._pre_rewrite_result

    def rewrite(self, original_model, rewritten_model):
      assert original_model == self._expected_original_model
      assert rewritten_model == self._expected_rewritten_model
      self.rewrite_called = True
      return self._rewrite_result

    def post_rewrite_validate(self, rewritten_model):
      assert rewritten_model == self._expected_rewritten_model
      self.post_rewrite_validate_called = True
      return self._post_rewrite_result

  def setUp(self):
    super(PerformRewriteTest, self).setUp()
    src_model_path = '/path/to/src/model'
    dst_model_path = '/path/to/dst/model'
    self._source_model = rewriter.ModelDescription(
        rewriter.ModelType.SAVED_MODEL, src_model_path)

    self._dest_model = rewriter.ModelDescription(
        rewriter.ModelType.SAVED_MODEL, dst_model_path)

  def testPerformRewriteCallsAllValidationsAndRewrites(self):
    rw = self.MyRewriter(True, True, True, self._source_model, self._dest_model)
    self.assertTrue(rewriter.perform_rewrite(
        self._source_model, self._dest_model, rw))
    self.assertTrue(rw.pre_rewrite_validate_called)
    self.assertTrue(rw.rewrite_called)
    self.assertTrue(rw.post_rewrite_validate_called)

  def testPerformRewriteStopsOnFailedPreRewriteValidation(self):
    rw = self.MyRewriter(
        False, True, True, self._source_model, self._dest_model)
    self.assertFalse(rewriter.perform_rewrite(
        self._source_model, self._dest_model, rw))
    self.assertTrue(rw.pre_rewrite_validate_called)
    self.assertFalse(rw.rewrite_called)
    self.assertFalse(rw.post_rewrite_validate_called)

  def testPeformRewriteStopsOnFailedRewrite(self):
    rw = self.MyRewriter(
        True, False, True, self._source_model, self._dest_model)
    self.assertFalse(rewriter.perform_rewrite(
        self._source_model, self._dest_model, rw))
    self.assertTrue(rw.pre_rewrite_validate_called)
    self.assertTrue(rw.rewrite_called)
    self.assertFalse(rw.post_rewrite_validate_called)

  def testPerformRewriteStopsOnFailedPostRewriteValidation(self):
    rw = self.MyRewriter(
        True, True, False, self._source_model, self._dest_model)
    self.assertFalse(rewriter.perform_rewrite(
        self._source_model, self._dest_model, rw))
    self.assertTrue(rw.pre_rewrite_validate_called)
    self.assertTrue(rw.rewrite_called)
    self.assertTrue(rw.post_rewrite_validate_called)


if __name__ == '__main__':
  absltest.main()
