{--
Copyright (C) 2022  Joel Berkeley

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
