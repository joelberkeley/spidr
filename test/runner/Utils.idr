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
