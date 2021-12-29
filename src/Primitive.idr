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
||| This module defines supported primitive backend types and their interaction with Idris.
module Primitive

import XLA.Literal

||| A `Primitive` is an Idris type for which there is a corresponding backend primitive type.
export
interface XLAPrimitive dtype => Primitive dtype where

export Primitive Bool where
export Primitive Int where
export Primitive Double where
