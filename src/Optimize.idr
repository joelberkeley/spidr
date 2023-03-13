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
||| This module contains definitions of function optimizers.
module Optimize

import Tensor

||| An `Optimizer` finds the value, in a `Tensor`-valued feature space, which (approximately)
||| optimizes a scalar-valued function over that space.
|||
||| If the function is not well-defined at points in the feature space, and this is expressed by
||| wrapping function values in some context, this extra context can be captured in the value `m`.
||| For example, a function `a -> Maybe (Tensor [] Double)`, can be optimized by an
||| `Optimizer {m=Maybe} a`.
|||
||| @domain The type of the domain over which to find the optimizer.
public export 0
Optimizer : {default id 0 m : Type -> Type} -> (0 domain : Type) -> Type
Optimizer a = (a -> m $ Ref $ Tensor [] F64) -> m a

||| Construct an `Optimizer` that implements grid search over a scalar feature space. Grid search
||| approximates the optimum by evaluating the objective over a finite, evenly-spaced grid.
|||
||| **NOTE** This function is not yet implemented.
|||
||| @density The density of the grid.
||| @lower The lower (inclusive) bound of the grid.
||| @upper The upper (exclusive) bound of the grid.
export
gridSearch : (density : Tensor [d] U32) ->
             (lower : Tensor [d] F64) ->
             (upper : Tensor [d] F64) ->
             Optimizer $ Ref $ Tensor [d] F64

||| The limited-memory BFGS (L-BFGS) optimization tactic, see
|||
||| Nocedal, Jorge, Updating quasi-Newton matrices with limited storage.
||| Math. Comp. 35 (1980), no. 151, 773–782.
|||
||| available at
|||
||| https://www.ams.org/journals/mcom/1980-35-151/S0025-5718-1980-0572855-7/
|||
||| **NOTE** This function is not yet implemented.
|||
||| @initialPoints The points from which to start optimization.
export
lbfgs : (initialPoints : Tensor [n] F64) -> Optimizer $ Ref $ Tensor [n] F64
