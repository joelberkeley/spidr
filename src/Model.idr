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

import public Model.GaussianProcess as Model
import public Model.Kernel as Model
import public Model.MeanFunction as Model

import Distribution
import Tensor

||| A `ProbabilisticModel` is a mapping from a feature space to a probability distribution over
||| a target space.
|||
||| @features The shape of the feature domain.
public export 0
ProbabilisticModel : forall marginal . Distribution targets marginal => (0 features : Shape {rank}) -> Type
ProbabilisticModel features = forall n . Tensor ((::) {len=rank} {elem=Nat} n features) Double -> marginal n
