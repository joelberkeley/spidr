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

import Device
import Tensor
import Model.Kernel

import Utils.Comparison
import Utils.Cases

rbfMatchesTFP : Device => Property
rbfMatchesTFP = fixedProperty $ do
  let lengthScale = tensor 0.4
      x = tensor [[-1.2], [-0.5], [0.3], [1.2]]
      x' = tensor [[-1.2], [-0.2], [0.8]]
      -- calculated with tensorflow probability
      -- >>> rbf = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=tf.cast(0.4, tf.float64))
      -- >>> x = [[-1.2], [-0.5], [0.3], [1.2]]
      -- >>> x_ = [[-1.2], [-0.2], [0.8]]
      -- >>> rbf.tensor(x, x_, x1_example_ndims=1, x2_example_ndims=1)
      expected = tensor [
          [1.00000000e+00, 4.39369377e-02, 3.72665456e-06],
          [2.16265177e-01, 7.54839608e-01, 5.08607003e-03],
          [8.83826492e-04, 4.57833372e-01, 4.57833372e-01],
          [1.52299879e-08, 2.18749152e-03, 6.06530669e-01]
        ]
  rbf lengthScale x x' ===# pure expected

export
group : Device => Group
group = MkGroup "Kernel" $ [
    ("rbf matches tfp", rbfMatchesTFP)
  ]
