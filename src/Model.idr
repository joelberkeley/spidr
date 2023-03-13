{--
Copyright 2021 Joel Berkeley

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
    marginalise : model -> {n : _} -> Tensor (S n :: features) F64 -> Ref $ marginal targets (S n)
