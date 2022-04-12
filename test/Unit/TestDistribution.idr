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

import Utils.Example
import Utils.Property

test_gaussian_pdf : Property
test_gaussian_pdf = withTests 1 $ property $ do
  let
    assertForUnivariate : Monad m =>
                          Literal [] Double -> Literal [] Double -> Literal [] Double -> TestT m ()
    assertForUnivariate mean cov x =
      let gaussian = MkGaussian (fromLiteral [[mean]]) (fromLiteral [[[cov]]])
          actual = pdf gaussian (fromLiteral [[x]])
          expected = fromLiteral [| univariate x mean cov |]
          msg = "Gaussian pdf mean \{show mean} cov \{show cov} x \{show x}"
       in fpTensorEq actual expected

          where
          univariate : Double -> Double -> Double -> Double
          univariate x mean cov = exp (- (x - mean) * (x - mean) / (2 * cov)) / sqrt (2 * pi * cov)

  sequence_ [assertForUnivariate mean cov x |
    mean <- [-2, -1, 0, 1, 2],
    cov <- [0.1, 1, 2],
    x <- the (List _) [-2, -1, 0, 1, 2]
  ]

  let mean = fromLiteral [[-0.2], [0.3]]
      cov = fromLiteral [[[1.2], [0.5]], [[0.5], [0.7]]]
      x = fromLiteral [[1.1], [-0.5]]
      actual = pdf (MkGaussian mean cov) x
      expected = fromLiteral 0.016427375
  fpTensorEq {tol=0.00000001} actual expected

test_gaussian_cdf : Property
test_gaussian_cdf = withTests 1 $ property $ do
  let gaussian = MkGaussian (fromLiteral [[0.5]]) (fromLiteral [[[1.44]]])
      xs : Vect _ _ = [-1.5, -0.5, 0.5, 1.5]
      expected = [0.04779036, 0.20232838, 0.5, 0.7976716]

      assert' : Monad m => (Literal [] Double, Literal [] Double) -> TestT m ()
      assert' (x, exp) =
        fpTensorEq {tol=0.0001} (cdf gaussian (fromLiteral [[x]])) (fromLiteral exp)

  traverse_ assert' (zip xs expected)

export
group : Group
group = MkGroup "Distribution" $ [
      ("Gaussian pdf", test_gaussian_pdf)
    , ("Gaussian cdf", test_gaussian_cdf)
  ]
