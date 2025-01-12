{--
Copyright (C) 2025  Joel Berkeley

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
