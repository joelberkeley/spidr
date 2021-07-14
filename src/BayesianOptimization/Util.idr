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
module BayesianOptimization.Util

import Distribution
import Tensor
import Model

||| Objective query points and either corresponding objective values or metadata.
public export 0
Data : {0 samples : Nat} -> Shape -> Shape -> Type
Data features targets =
  (Tensor (samples :: features) Double, Tensor (samples :: targets) Double)

infix 9 ~>

||| An `i ~> o` is a minimal wrapper for unary functions.
public export
record (~>) i o where
  constructor MkUnary
  run : (i -> o)

export
Functor (i ~>) where
  map f (MkUnary g) = MkUnary (f . g)

export
Applicative (i ~>) where
  pure x = MkUnary (\_ => x)
  (MkUnary f) <*> (MkUnary g) = MkUnary (\i => (f i) (g i))

export
Monad (i ~>) where
  join x = MkUnary (\i => run (run x i) i)
