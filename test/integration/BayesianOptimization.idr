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
import Tensor
import Model
import Distribution
import BayesianOptimization
import Optimize
import Util

historic_data : Data {samples=5} [2] [1]

public export 0 Model : Type
Model = ProbabilisticModel [2] {targets=[1]} {marginal=Gaussian [1]}

model : Model

optimizer : AcquisitionOptimizer

-- todo we definitely shouldn't need to specify the sample and batch sizes
new_point : Maybe $ Tensor [1, 2] Double
new_point = let ei = direct $ expectedImprovementByModel {s=_}
                acquisition = map optimizer ei
             in run acquisition (historic_data, model)

record DataPair o f where
  constructor MkDataPair
  objective : o
  failure : f

new_point_constrained : Maybe $ Tensor [1, 2] Double
new_point_constrained = let eci = objective >>> expectedConstrainedImprovement {s=_}
                            pof = failure >>> (probabilityOfFeasibility $ const 0.5) {s=_}
                            acquisition = map optimizer (eci <*> pof)
                            dataAndModel = MkDataPair (historic_data, model) (historic_data, model)
                         in run acquisition dataAndModel
