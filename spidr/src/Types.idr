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

public export 0
NameMe : Type -> Type
NameMe a = HasIO io => Show e => EitherT io e a

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
