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

import Literal
import Tensor
import Distribution

import Utils.Comparison
import Utils.Cases

covering
gaussianUnivariatePDF : Property
gaussianUnivariatePDF = property $ do
  let doubles = literal [] doubles
  [mean, cov, x] <- forAll (np [doubles, doubles, doubles])
  let gaussian = MkGaussian (fromLiteral [[mean]]) (fromLiteral [[[cov]]])
      actual = pdf gaussian (fromLiteral [[x]])
      expected = fromLiteral [| univariate x mean cov |]
  actual ===? expected

    where
    univariate : Double -> Double -> Double -> Double
    univariate x mean cov = exp (- (x - mean) * (x - mean) / (2 * cov)) / sqrt (2 * pi * cov)

gaussianMultivariatePDF : Property
gaussianMultivariatePDF = withTests 1 $ property $ do
  let mean = fromLiteral [[-0.2], [0.3]]
      cov = fromLiteral [[[1.2], [0.5]], [[0.5], [0.7]]]
      x = fromLiteral [[1.1], [-0.5]]
  pdf (MkGaussian mean cov) x ===? 0.016427375

gaussianCDF : Property
gaussianCDF = withTests 1 $ property $ do
  let gaussian = MkGaussian (fromLiteral [[0.5]]) (fromLiteral [[[1.44]]])

  cdf gaussian (fromLiteral [[-1.5]]) ===? 0.04779036
  cdf gaussian (fromLiteral [[-0.5]]) ===? 0.20232838
  cdf gaussian (fromLiteral [[0.5]]) ===? 0.5
  cdf gaussian (fromLiteral [[1.5]]) ===? 0.7976716

export covering
group : Group
group = MkGroup "Distribution" $ [
      ("Gaussian univariate pdf", gaussianUnivariatePDF)
    , ("Gaussian multivariate pdf", gaussianMultivariatePDF)
    , ("Gaussian cdf", gaussianCDF)
  ]
