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
module BayesianOptimization

import Tensor
import Data.Vect
import Optimize
import Distribution

interface (Distribution samples targets dist) => ProbabilisticModel (0 features : Shape) (0 targets : Shape) dist model where
  predict : model -> Tensor (samples :: features) Double -> dist

interface Domain where

Acquisition : Nat -> Shape -> Type
Acquisition batch_size features = Tensor (batch_size :: features) Double -> Tensor [] Double

AcquisitionOptimizer : {batch_size : Nat} -> {features : Shape} -> Type
AcquisitionOptimizer = Optimizer $ Tensor (batch_size :: features) Double

public export
Data : Shape -> Shape -> Type
Data features targets = (Tensor features Double, Tensor targets Double)

public export
AcquisitionBuilder : (ProbabilisticModel features targets model dist) => {model : Type} -> (features : Shape) -> (targets : Shape) -> Nat -> Type
AcquisitionBuilder {model} features targets batch_size = Data features targets -> model -> Acquisition batch_size features

expected_improvement : (ProbabilisticModel features [] (Gaussian samples [1]) model) => model -> (best : Tensor [] Double) -> Acquisition 1 features
expected_improvement model best at = let normal = predict model at in (best - mean normal) * (cdf normal best) + (?squeeze $ covariance normal) * ?prob

--expected_improvement_by_model : (ProbabilisticModel features [] model) => Data features [] -> model -> Acquisition 1 features
--expected_improvement_by_model (query_points, _) model' at = let best = min $ predict model' (?expand_dims0 query_points) in expected_improvement model best
