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
||| For internal spidr use only.
module Compiler.StableHLO.Dialect.Version

import Compiler.FFI

export
data Version = MkVersion GCAnyPtr

%foreign (libxla "Version_delete")
prim__delete : AnyPtr -> PrimIO ()

%foreign (libxla "Version_getMinimumVersion")
prim__versionGetMinimumVersion : PrimIO AnyPtr

export
getMinimumVersion : HasIO io => io Version
getMinimumVersion = do
  version <- primIO prim__versionGetMinimumVersion
  version <- onCollectAny version (primIO . prim__delete)
  pure (MkVersion version)

%foreign (libxla "Version_toString")
prim__versionToString : GCAnyPtr -> PrimIO AnyPtr

export
toString : HasIO io => Version -> io CppString
toString (MkVersion version) = do
  str <- primIO $ prim__versionToString version
  str <- onCollectAny str (primIO . prim__stringDelete)
  pure (MkCppString str)
