{--
Copyright 2022 Joel Berkeley

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
module Unit.TestOptimize

import Literal
import Optimize
import Tensor

import Utils.Cases
import Utils.Comparison

gridSearch : Property
gridSearch = fixedProperty $ do
  let lower = fromLiteral [-1.0, -1.0]
      upper = fromLiteral [1.0, 1.0]
      offset = fromLiteral [0.2, -0.1]

      f : Tensor [2] F64 -> Tensor [] F64
      f x = reduce @{Sum} [0] $ (x - offset) ^ fill 2.0

  gridSearch [100, 100] lower upper f ===# offset

export covering
group : Group
group = MkGroup "Optimize" $ [
    ("grid search", gridSearch)
  ]
