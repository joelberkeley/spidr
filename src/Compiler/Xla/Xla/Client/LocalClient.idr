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
module Compiler.Xla.Xla.Client.LocalClient

import Compiler.Xla.Prim.Xla.Client.LocalClient
import Compiler.Xla.Xla.Literal
import Compiler.Xla.Xla.Client.XlaComputation

public export
data LocalClient : Type where
  MkLocalClient : AnyPtr -> LocalClient

export
executeAndTransfer : HasIO io => LocalClient -> XlaComputation -> io Literal
executeAndTransfer (MkLocalClient client) (MkXlaComputation computation) = do
  literal <- primIO $ prim__executeAndTransfer client computation prim__getNullAnyPtr 0
  literal <- onCollectAny literal Literal.delete
  pure (MkLiteral literal)
