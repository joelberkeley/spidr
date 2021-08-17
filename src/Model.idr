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

||| Observed pairs of data points from feature and target domains. Data sets such as this are
||| commonly used in supervised learning settings.
|||
||| @samples The number of points in each of the feature and target data.
||| @features The shape of the feature domain.
||| @targets The shape of the target domain.
public export 0
Data : {0 samples : Nat} -> (0 features : Shape) -> (0 targets : Shape) -> Type
Data features targets =
  (Tensor (Vect.(::) samples features) Double, Tensor (Vect.(::) samples targets) Double)

||| A `ProbabilisticModel` is a mapping from a feature domain to a probability distribution over
||| a target domain.
|||
||| @features The shape of the feature domain.
||| @targets The shape of the target domain.
||| @marginal The type of mulitvariate marginal distribution over the target domain.
public export 0
ProbabilisticModel : Distribution targets marginal => (0 features : Shape) -> Type
ProbabilisticModel features = {n : _} -> Tensor (Vect.(::) (S n) features) Double -> marginal (S n)
