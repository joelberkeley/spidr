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
import Optimize
import Tensor

||| Objective query points and either corresponding objective values or metadata.
|||
||| @samples The number of points in each of the feature and target data.
||| @features The shape of the feature domain.
||| @targets The shape of the target domain.
public export 0
Data : {0 samples : Nat} -> (0 features : Shape) -> (0 targets : Shape) -> Type
Data features targets = (Tensor (samples :: features) Double, Tensor (samples :: targets) Double)

||| A `ProbabilisticModel` is a mapping from a feature space to a probability distribution over
||| a target space.
|||
||| @features The shape of the feature domain.
||| @targets The shape of the target domain.
||| @marginal The type of mulitvariate marginal distribution.
public export
interface Distribution targets marginal =>
 ProbabilisticModel (0 features : Shape) (0 targets : Shape {rank=r})
    (0 marginal : Multivariate {rank=r}) model | model where
  marginalise : model -> {s : _} -> Tensor ((S s) :: features) Double -> marginal targets (S s)

||| @features The shape of the feature domain.
||| @targets The shape of the target domain.
public export
interface Trainable (0 features : Shape) (0 targets : Shape) ty where
  fit : ty
    -> (forall n . Tensor [n] Double -> Optimizer $ Tensor [n] Double)
    -> {s : _} -> Data {samples=S s} features targets
    -> ty
