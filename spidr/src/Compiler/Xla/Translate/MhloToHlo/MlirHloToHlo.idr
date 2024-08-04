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
module Compiler.Xla.Translate.MlirHloToHlo

%foreign (libxla "ConvertMlirHloToHloModule")
prim__convertMlirHloToHloModule : AnyPtr -> PrimIO AnyPtr

export
convertMlirHloToHloModule : HasIO io => ModuleOp -> io XlaComputation
convertMlirHloToHloModule (MkModuleOp moduleOp) = do
  computation <- primIO $ prim__convertHloToMlirHlo moduleOp
  computation <- onCollectAny computation XlaComputation.delete
  pure (MkXlaComputation computation)
