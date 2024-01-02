{--
Copyright 2024 Joel Berkeley

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
||| This module extends the standard library's Control.Function
module Control.Local.Function

||| A dependent variant of `flip` where the return type can depend on the input values. `dflip`
||| flips the order of arguments for a function, such that `dflip f x y` is the same as `f y x`.
public export
dflip : {0 c : a -> b -> Type} -> ((x : a) -> (y : b) -> c x y) -> (y : b) -> (x : a) -> c x y
dflip f y x = f x y
