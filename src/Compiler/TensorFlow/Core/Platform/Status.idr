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
module Compiler.TensorFlow.Core.Platform.Status

import System.FFI

import Compiler.FFI

public export
data Status : Type where
  MkStatus : GCAnyPtr -> Status

%foreign (libxla "Status_delete")
prim__delete : AnyPtr -> PrimIO ()

export
delete : AnyPtr -> IO ()
delete = primIO . prim__delete

%foreign (libxla "Status_ok")
prim__ok : GCAnyPtr -> Int

export
ok : Status -> Bool
ok (MkStatus ptr) = cIntToBool (prim__ok ptr)
