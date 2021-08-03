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

import public Data.Morphisms
import Distribution
import Tensor
import Model

infix 9 >>>

||| Compose two functions that each use two values and wrap them in a morphism. This is a
||| convenience function for contructing unary wrappers with `Empiric`s and the corresponding
||| handler functions for data and models.
export
(>>>) : (i -> (a, b)) -> (a -> b -> o) -> i ~> o
f >>> g = Mor (uncurry g . f)

export
run : (i ~> o) -> i -> o
run = applyMor
