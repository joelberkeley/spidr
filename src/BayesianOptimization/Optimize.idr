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
module BayesianOptimization.Optimize

import BayesianOptimization.Domain
import Tensor
import Optimize

||| Construct an `Optimizer` that implements grid search over a feature space. Grid search
||| approximates the optimum by evaluating the objective over a finite, evenly-spaced grid.
|||
||| @density The density of the grid.
||| @lower The lower (inclusive) bound of the grid.
||| @upper The upper (exclusive) bound of the grid.
export
gridSearch : (density : Tensor features Integer) ->
             (domain : ContinuousDomain features) ->
             Optimizer (Tensor (1 :: features) Double)
