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
module Utils.TestComparison

import Utils.Cases
import Utils.Comparison

sufficientlyEqCases : List (Double, Double)
sufficientlyEqCases = [
  (0.0, 0.0),
  (1.1, 1.1),
  (1.1, 1.1 + floatingPointTolerance / 2),
  (1.1, 1.1 - floatingPointTolerance / 2),
  (-1.1, -1.1),
  (-1.1, -1.1 + floatingPointTolerance / 2),
  (-1.1, -1.1 - floatingPointTolerance / 2),
  (inf, inf),
  (-inf, -inf),
  (nan, nan)
]

insufficientlyEqCases : List (Double, Double)
insufficientlyEqCases =
  let cases = [
      (0.0, floatingPointTolerance / 2),
      (floatingPointTolerance / 2, 0.0),
      (0.0, - floatingPointTolerance / 2),
      (- floatingPointTolerance / 2, 0.0),
      (1.1, 1.1 + floatingPointTolerance * 2),
      (1.1, 1.1 - floatingPointTolerance * 2),
      (-1.1, -1.1 + floatingPointTolerance * 2),
      (-1.1, -1.1 - floatingPointTolerance * 2),
      (0.0, inf),
      (1.1, inf),
      (-1.1, inf),
      (0.0, -inf),
      (1.1, -inf),
      (-1.1, -inf),
      (0.0, nan),
      (1.1, nan),
      (-1.1, nan),
      (inf, -inf),
      (inf, nan),
      (-inf, nan)
  ] in cases ++ map (\(x, y) => (y, x)) cases

isSufficientlyEq : Property
isSufficientlyEq = withTests 1 $ property $ do
  traverse_ (\(x, y) => diff x sufficientlyEq y) sufficientlyEqCases

isNotSufficientlyEq : Property
isNotSufficientlyEq = withTests 1 $ property $ do
  traverse_ (\(x, y) => diff x (not .: sufficientlyEq) y) insufficientlyEqCases

export
group : Group
group = MkGroup "Test comparison utilities" [
      ("sufficientlyEq", isSufficientlyEq)
    , ("not sufficientlyEq", isNotSufficientlyEq)
  ]
