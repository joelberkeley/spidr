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

import Control.Monad.Identity
import public Control.Monad.Reader
import public Data.Stream

import public BayesianOptimization.Acquisition as BayesianOptimization
import Tensor

infix 9 >>>

||| Compose two functions that each use two values and wrap them in a `Reader`. This is a
||| convenience function for contructing unary wrappers with `Empiric`s and the corresponding
||| handler functions for data and models.
export
(>>>) : (i -> (a, b)) -> (a -> b -> o) -> Reader i o
f >>> g = MkReaderT (Id . uncurry g . f)

||| A `Stream`-like collection where each successive element is wrapped in an additional `Ref`.
public export
data RefStream : Type -> Type where
  (::) : a -> Inf (Ref (RefStream a)) -> RefStream a

public export
take : (n : Nat) -> RefStream a -> Ref $ Vect n a
take Z _ = pure Nil
take (S k) (x :: xs) = pure (x :: !(take k !xs))

||| A Bayesian optimization loop as a (potentially infinite) stream of values. The values are
||| typically the observed data, and the models of that data. The loop iteratively finds new points
||| with the specified `tactic` then updates the values with these new points (assuming some
||| implicit objective function).
|||
||| @tactic The tactic which which to recommend new points. This could be optimizing an acquisition
|||   function, for example. Note this is a `Morphism`, not a function.
||| @observer A function which evaluates the optimization objective at the recommended points, then
|||   updates the values (typically data and models).
export covering
loop :
  (tactic : Reader env $ Ref $ Tensor shape dtype) ->
  (observer : Tensor shape dtype -> env -> Ref env) ->
  env ->
  Ref $ RefStream env
loop tactic observer env = do
  env' <- observer !(runReader env tactic) env
  pure (env' :: loop tactic observer env')
