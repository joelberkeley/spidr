{--
Copyright 2022 Joel Berkeley

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
--}
module Unit.TestDistribution

import Utils

import Tensor
import Distribution

test_gaussian_pdf : IO ()
test_gaussian_pdf = do
  let
    assertMono : Double -> Double -> Double -> IO ()
    assertMono mean cov x =
      let gaussian = MkGaussian (const {shape=[1, 1]} [[mean]]) (const [[[cov]]])
          actual = pdf gaussian (const {shape=[1, 1]} [[x]])
          expected = const (exp (- (x - mean) * (x - mean) / (2 * cov)) / sqrt (2 * pi * cov))
          msg = "Gaussian mean \{show mean} cov \{show cov} x \{show x}"
       in assertAll msg (sufficientlyEq actual expected)

  sequence_ [assertMono mean cov x |
    mean <- [-2, -1, 0, 1, 2],
    cov <- [0.1, 1, 2],
    x <- the (List _) [-2, -1, 0, 1, 2]
  ]

  let mean = const {shape=[2, 1]} [[-0.2], [0.3]]
      cov = const [[[1.2], [0.5]], [[0.5], [0.7]]]
      x = const {shape=[2, 1]} [[1.1], [-0.5]]
      actual = pdf (MkGaussian mean cov) x
      expected = const 0.016427375  -- calculated using TensorFlow Probability
  assertAll "multivariate Gaussian" $ sufficientlyEq {tol=0.00000001} actual expected

export
test : IO ()
test = do
  test_gaussian_pdf
