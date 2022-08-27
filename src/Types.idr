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

import public Data.Vect

import Shape

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
Bounded Double where
  min = -1.7976931348623157e308
  max = 1.7976931348623157e308
