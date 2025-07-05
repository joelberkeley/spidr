{--
Copyright (C) 2021  Joel Berkeley

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
||| This module contains definitions and utilities for probabilistic models.
module Model

import Distribution
import Tensor

||| A `ProbabilisticModel` is a mapping from a feature domain to a probability distribution over
||| a target domain.
|||
||| @features The shape of the feature domain.
||| @targets The shape of the target domain.
||| @marginal The type of mulitvariate marginal distribution over the target domain.
public export
interface Distribution marginal => ProbabilisticModel
    (0 features, targets : Shape)
    (0 marginal : (0 event : Shape) -> (0 dim : Nat) -> Type)
    model | model
  where
    ||| Return the marginal distribution over the target domain at the specified feature values.
    marginalise : model -> {n : _} -> Tensor (S n :: features) F64 -> Tag $ marginal targets (S n)
