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
module Unit.BayesianOptimization.TestAcquisition

import Utils
import Utils.Cases
import Utils.Comparison

import Constants
import Literal
import Tensor
import Data
import Model
import Distribution
import BayesianOptimization.Acquisition

record LinearModel (d : Nat) where
  constructor MkLinearModel
  m : Tensor [S d] F64
  c : Tensor [] F64

ProbabilisticModel [S d] [1] Gaussian (LinearModel d) where
  marginalise (MkLinearModel m c) x = MkGaussian (expand 1 (x @@ m) + broadcast c) (fill 0.5)

-- we can fairly easily make this not fixed, no?
expectedConstrainedImprovement : Property
expectedConstrainedImprovement = fixedProperty $ do
  let name = "expected constrained improvement is expected improvement for no constraint"
      dataset = MkDataset (fromLiteral [[0.0]]) (fromLiteral [[0.0]])
      model = MkLinearModel (fromLiteral [0.0]) 1.0
      x = fromLiteral [[1.0]]
      eci = expectedConstrainedImprovement (-inf) dataset model
      ei = expectedImprovementByModel dataset model

  eci (const 1.0) x ===# ei x

export covering
group : Group
group = MkGroup "Acquisition" $ [
    ("expected constrained improvement", expectedConstrainedImprovement)
  ]
