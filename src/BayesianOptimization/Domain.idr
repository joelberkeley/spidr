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
module BayesianOptimization.Domain

import Tensor

||| A continuous space where each dimension is bounded by a constant lower and upper bound.
public export
record ContinuousDomain {rank : Nat} (shape : Shape {rank=rank}) where
  constructor MkContinuousDomain
  -- todo possible to prove upper > lower?
  lower : Tensor shape Double
  upper : Tensor shape Double
