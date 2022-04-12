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
module Unit.Model.TestKernel

import Literal
import Tensor
import Model.Kernel

import Utils.Property

covering
rbfMatchesTFP : Property
rbfMatchesTFP = withTests 1 $ property $ do
  let length_scale = fromLiteral 0.4
      x = fromLiteral [[-1.2], [-0.5], [0.3], [1.2]]
      x' = fromLiteral [[-1.2], [-0.2], [0.8]]
      expected = fromLiteral [  -- calculated with tensorflow probability
          [       1.0, 0.04393695, 0.00000373],
          [0.21626519, 0.75483966, 0.00508606],
          [0.00088383, 0.45783338, 0.45783338],
          [       0.0, 0.00218749, 0.60653049]
        ]
  fpTensorEq {tol=0.000001} (rbf length_scale x x') expected

export covering
group : Group
group = MkGroup "Kernel" $ [
    ("rbf matches tfp", rbfMatchesTFP)
  ]
