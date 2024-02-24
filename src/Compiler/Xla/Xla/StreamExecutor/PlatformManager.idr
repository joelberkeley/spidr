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
module Compiler.Xla.Xla.StreamExecutor.PlatformManager

import Compiler.Xla.Prim.Xla.StreamExecutor.PlatformManager
import Compiler.Xla.Xla.Status
import Compiler.Xla.Xla.StreamExecutor.Platform

export
registerPlatform : HasIO io => Platform -> io Status
registerPlatform (MkPlatform platform) = do
  status <- primIO $ prim__registerPlatform platform
  status <- onCollectAny status Status.delete
  pure (MkStatus status)

export
platformWithName : HasIO io => String -> io Platform
platformWithName target = do
  platform <- primIO $ prim__platformWithName target
  pure (MkPlatform platform)
