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
||| Functionality for Bayesian optimization, the data-efficient optimization of objective functions.
||| Bayesian optimization recommends new points at which to query your objective by placing a
||| probabilistic model over historic data then, typically, optimizing an _acquisition function_
||| which quantifies how useful it would be to evaluate the objective at any given set of points.
module BayesianOptimization

import Control.Monad.Reader

import public BayesianOptimization.Acquisition as BayesianOptimization
import Data
import Model
import Tensor

||| A `Stream`-like collection where each successive element can extend the set of `Tag`s.
public export
data TagStream : Type -> Type where
  (::) : a -> Inf (Tag (TagStream a)) -> TagStream a

||| Take `n` values from a `TagStream`, sequencing the `Tag` effects.
public export
take : (n : Nat) -> TagStream a -> Tag $ Vect n a
take Z _ = pure Nil
take (S k) (x :: xs) = pure (x :: !(take k !xs))

||| Create an infinite stream of values from a generator function and a starting value.
export covering
iterate : (a -> Tag a) -> a -> Tag $ TagStream a
iterate f x = do
  x' <- f x
  pure (x' :: iterate f x')

||| Construct a single simple Bayesian optimization step.
|||
||| @objective The objective function to optimize.
||| @train Used to train the model on new data.
||| @tactic The tactic, such as an optimized acquisition function, to find a new point from the
|||   data and model
export
step : (objective : forall n . Tensor (n :: features) F64 -> Tag $ Tensor (n :: targets) F64) ->
       (probabilisticModel : ProbabilisticModel features targets marginal model) =>
       (train : Dataset features targets -> model -> Tag $ model) ->
       (tactic : ReaderT (DataModel {probabilisticModel} model) Tag (Tensor (1 :: features) F64)) ->
       DataModel {probabilisticModel} model ->
       Tag $ DataModel {probabilisticModel} model
step objective train tactic env = do
  newPoint <- runReaderT env tactic
  dataset <- tag $ concat env.dataset $ MkDataset newPoint !(objective newPoint)
  pure $ MkDataModel !(train dataset env.model) dataset
