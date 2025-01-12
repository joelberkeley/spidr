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
module Utils.TestComparison

import Utils.Cases
import Utils.Comparison

-- compilation is very slow without these
%hide Utils.Cases.Literal.inf
%hide Utils.Cases.Literal.nan

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
isSufficientlyEq = fixedProperty $ do
  traverse_ (\(x, y) => diff x sufficientlyEq y) sufficientlyEqCases

isNotSufficientlyEq : Property
isNotSufficientlyEq = fixedProperty $ do
  traverse_ (\(x, y) => diff x (not .: sufficientlyEq) y) insufficientlyEqCases

export
group : Group
group = MkGroup "Test comparison utilities" [
      ("sufficientlyEq", isSufficientlyEq)
    , ("not sufficientlyEq", isNotSufficientlyEq)
  ]
