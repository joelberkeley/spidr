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

import Literal
import Tensor

||| An `Optimizer` finds the value, in a `Tensor`-valued feature space, which (approximately)
||| optimizes a scalar-valued function over that space.
|||
||| @domain The shape of a single point in the domain.
public export 0
Optimizer : (domain : Shape) -> Type
Optimizer domain = (Tensor domain F64 -> Ref $ Tensor [] F64) -> Ref $ Tensor domain F64

||| Grid search over a scalar feature space. Grid search approximates the optimum by evaluating the
||| objective over a finite, evenly-spaced grid.
|||
||| @density The density of the grid.
||| @lower The lower (inclusive) bound of the grid.
||| @upper The upper (exclusive) bound of the grid.
export
gridSearch : {d : _} ->
             (density : Vect d Nat) ->
             (lower : Tensor [d] F64) ->
             (upper : Tensor [d] F64) ->
             Optimizer [d]
gridSearch {d=Z} _ _ _ _ = fromLiteral []
gridSearch {d=S k} density lower upper f =
  let densityAll : Nat
      densityAll = product density

      prodDims : Tensor [S k] U64 := fromLiteral $ cast $ scanr (*) 1 (tail density)
      idxs = fromLiteral {shape=[densityAll]} $ cast $ Vect.range densityAll
      densityTensor = broadcast $ fromLiteral {shape=[S k]} {dtype=U64} (cast density)
      grid = broadcast {to=[densityAll, S k]} (expand 1 idxs)
        `div` broadcast {from=[S k]} (cast prodDims) `rem` densityTensor
      gridRelative = cast grid / cast densityTensor
      points = with Tensor.(+)
        broadcast lower + broadcast {to=[densityAll, _]} (upper - lower) * gridRelative
   in slice [at (argmin 0 (vmap f points))] points

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
lbfgs : (initialPoints : Tensor [n] F64) -> Optimizer [n]
