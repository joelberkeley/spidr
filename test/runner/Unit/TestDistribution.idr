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

import Device
import Tensor
import Distribution

import Utils.Comparison
import Utils.Cases

gaussianUnivariatePDF : Device => Property
gaussianUnivariatePDF = property $ do
  let doubles = literal [] doubles
  [mean, cov, x] <- forAll (np [doubles, doubles, doubles])
  let gaussian = MkGaussian (tensor [[mean]]) (tensor [[[cov]]])
      actual = pdf gaussian (tensor [[x]])
      expected = tensor [| univariate x mean cov |]
  actual ===# pure expected

    where
    univariate : Double -> Double -> Double -> Double
    univariate x mean cov = exp (- (x - mean) * (x - mean) / (2 * cov)) / sqrt (2 * pi * cov)

gaussianMultivariatePDF : Device => Property
gaussianMultivariatePDF = fixedProperty $ do
  let mean = tensor [[-0.2], [0.3]]
      cov = tensor [[[1.2], [0.5]], [[0.5], [0.7]]]
      x = tensor [[1.1], [-0.5]]
  pdf (MkGaussian mean cov) x ===# pure 0.016427375

gaussianCDF : Device => Property
gaussianCDF = fixedProperty $ do
  let gaussian = MkGaussian (tensor [[0.5]]) (tensor [[[1.44]]])

  cdf gaussian (tensor [[-1.5]]) ===# pure 0.04779036
  cdf gaussian (tensor [[-0.5]]) ===# pure 0.20232838
  cdf gaussian (tensor [[0.5]]) ===# pure 0.5
  cdf gaussian (tensor [[1.5]]) ===# pure 0.7976716

export
group : Device => Group
group = MkGroup "Distribution" $ [
      ("Gaussian univariate pdf", gaussianUnivariatePDF)
    , ("Gaussian multivariate pdf", gaussianMultivariatePDF)
    , ("Gaussian cdf", gaussianCDF)
  ]
