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
module XLA.Client.XlaComputation

import System.FFI

import XLA.FFI
import XLA.Literal

public export
XlaComputation : Type
XlaComputation = Struct "XlaComputation" []

%foreign (libxla "XlaComputation_delete")
prim__XlaComputation_delete : XlaComputation -> PrimIO ()

export
delete : XlaComputation -> IO ()
delete = primIO . prim__XlaComputation_delete

export
%foreign (libxla "execute")
prim__execute : XlaComputation -> PrimIO Literal
