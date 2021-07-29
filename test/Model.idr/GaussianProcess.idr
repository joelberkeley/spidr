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
import Model
import Tensor
import Distribution
import Optimize

mkPrior : Tensor [2] Double -> GaussianProcess [2]
mkPrior params = MkGP zero (rbf $ index 0 params)

mkLikelihood : {d : _} -> Tensor [2] Double -> Gaussian [] (S d)
mkLikelihood params = MkGaussian (fill 0) (diag $ index 1 params)

data_ : (Tensor [20, 2] Double, Tensor [20] Double)

predictions : (Tensor [3] Double, Tensor [3, 3] Double)
predictions = let optimizer = gridSearch (const [100, 100]) (const [-2, -2]) (const [2, 2])
                  posterior = fit optimizer mkPrior mkLikelihood data_
                  marginal = marginalise posterior $ const {shape=[3, 2]} [[0, 0], [0, 1], [1, 0]]
               in (mean marginal, cov marginal)
