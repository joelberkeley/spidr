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
module Utils

import Device
import Tensor

export
[Finite] Bounded (Literal [] Double) where
  min = Scalar (min @{Finite})
  max = Scalar (max @{Finite})

export
isNan : Double -> Bool
isNan x = x /= x

namespace Tag
  export
  unsafeEval : Device => PrimitiveRW dtype ty => Tag (Tensor shape dtype) -> Literal shape ty
  unsafeEval @{device} = unsafePerformIO . eval device

export
unsafeEval : Device => PrimitiveRW dtype ty => Tensor shape dtype -> Literal shape ty
unsafeEval @{device} x = unsafePerformIO $ eval device (pure x)

namespace TensorList
  namespace Tag
    export
    unsafeEval : Device => Tag (TensorList shapes tys) -> All2 Literal shapes tys
    unsafeEval @{device} = unsafePerformIO . eval device

  export
  unsafeEval : Device => TensorList shapes tys -> All2 Literal shapes tys
  unsafeEval @{device} xs = unsafePerformIO $ eval device (pure xs)
