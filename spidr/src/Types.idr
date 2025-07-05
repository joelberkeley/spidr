{--
Copyright (C) 2021  Joel Berkeley

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
||| This module contains common library types.
module Types

import public Data.Nat
import public Data.Vect

||| Describes the shape of a `Tensor`. For example, a `Tensor` of `Double`s with contents
||| `[[0, 1, 2], [3, 4, 5]]` has two elements in its outer-most axis, and each of those elements
||| has three `Double`s in it, so this has shape [2, 3]. A `Tensor` can have axes of zero length,
||| though the shape cannot be unambiguously inferred by visualising it. For example, `[[], []]`
||| can have shape [2, 0], [2, 0, 5] or etc. A scalar `Tensor` has shape `[]`.
public export 0
Shape : Type
Shape = List Nat

||| An `Array shape dtype` is either:
||| 
||| * a single value of type `dtype` (for `shape` `[]`), or
||| * an arbitrarily nested array of `Vect`s of such values (for any other `shape`)
|||
||| @shape The shape of the array.
||| @dtype The type of elements of the array.
public export 0
Array : (0 shape : Shape) -> (0 dtype : Type) -> Type
Array [] dtype = dtype
Array (d :: ds) dtype = Vect d (Array ds dtype)

||| A type `a` satisfying `Bounded a` has a minimum and a maximum value.
public export
interface Bounded a where
  min : a
  max : a

export
Bounded Int32 where
  min = -2147483648
  max = 2147483647

export
[Finite] Bounded Double where
  min = -1.7976931348623157e308
  max = 1.7976931348623157e308
