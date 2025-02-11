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
||| Function optimizers.
module Optimize

import Tensor

||| An `Optimizer` finds the value, in a `Tensor`-valued feature space, which (approximately)
||| optimizes a scalar-valued function over that space.
|||
||| @domain The type of the domain over which to find the optimal value.
public export 0
Optimizer : (0 domain : Type) -> Type
Optimizer a = (a -> Tag $ Tensor [] F64) -> Tag a

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
             Optimizer $ Tensor [d] F64

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
lbfgs : (initialPoints : Tensor [n] F64) -> Optimizer $ Tensor [n] F64
