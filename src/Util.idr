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
||| This module contains miscellaneous utilities
module Util

||| The constant Pi
export
PI : Double
PI = 3.1415926535897932385

||| An `Error` indicates a runtime error.
public export
interface Error exc where
  ||| Show the error in a human readable format.
  format : exc -> String

||| Indicates a value is invalid.
public export
data ValueError = MkValueError String

export
Error ValueError where
  format (MkValueError msg) = msg
