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
public export 0
Optimizer : {default id m : Type -> Type} -> Type -> Type
Optimizer a = (a -> m $ Tensor [] Double) -> m a
