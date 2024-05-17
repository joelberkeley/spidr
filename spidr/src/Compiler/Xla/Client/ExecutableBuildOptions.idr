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
module Compiler.Xla.Client.ExecutableBuildOptions

import Compiler.FFI

public export
data ExecutableBuildOptions = MkExecutableBuildOptions AnyPtr

%foreign (libxla "ExecutableBuildOptions_delete")
prim__delete : AnyPtr -> PrimIO ()

export
delete : HasIO io => ExecutableBuildOptions -> io ()
delete (MkExecutableBuildOptions opts) = primIO $ prim__delete opts

%foreign (libxla "ExecutableBuildOptions_new")
prim__mkExecutableBuildOptions : PrimIO AnyPtr

export
mkExecutableBuildOptions : HasIO io => io ExecutableBuildOptions
mkExecutableBuildOptions = MkExecutableBuildOptions <$> primIO prim__mkExecutableBuildOptions
