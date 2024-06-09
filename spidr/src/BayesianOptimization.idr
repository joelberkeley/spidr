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
||| This module defines functionality for Bayesian optimization, the data-efficient optimization of
||| objective functions. Bayesian optimization recommends new points at which to query your
||| objective by placing a probabilistic model over historic data then, typically, optimizing an
||| _acquisition function_ which quantifies how useful it would be to evaluate the objective at any
||| given set of points.
module BayesianOptimization

import Control.Monad.Reader

import public BayesianOptimization.Acquisition as BayesianOptimization
import Data
import Model
import Tensor

||| A `Stream`-like collection where each successive element extends the `Graph`.
public export
data GraphStream : Type -> Type where
  (::) : a -> Inf (Graph (GraphStream a)) -> GraphStream a

||| Take `n` values from a `GraphStream`, sequencing the `Graph` effects.
public export
take : (n : Nat) -> GraphStream a -> Graph $ Vect n a
take Z _ = pure Nil
take (S k) (x :: xs) = pure (x :: !(take k !xs))

||| Create an infinite stream of values from a generator function and a starting value.
export covering
iterate : (a -> Graph a) -> a -> Graph $ GraphStream a
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
step : (objective : forall n . Tensor (n :: features) F64 -> Graph $ Tensor (n :: targets) F64) ->
       (probabilisticModel : ProbabilisticModel features targets marginal model) =>
       (train : Dataset features targets -> model -> Graph $ model) ->
       (tactic : Reader (DataModel {probabilisticModel} model) (Graph $ Tensor (1 :: features) F64)) ->
       DataModel {probabilisticModel} model ->
       Graph $ DataModel {probabilisticModel} model
step objective train tactic env = do
  newPoint <- runReader env tactic
  let dataset = concat env.dataset $ MkDataset newPoint !(objective newPoint)
  pure $ MkDataModel !(train dataset env.model) dataset
