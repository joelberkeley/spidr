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
module Compiler.TensorFlow.Core.CommonRuntime.GPU.GPUInit

import Compiler.Foreign.TensorFlow.Core.CommonRuntime.GPU.GPUInit
import Compiler.TensorFlow.StreamExecutor.Platform
import Compiler.TensorFlow.Core.Platform.Status

export
validateGPUMachineManager : IO Status
validateGPUMachineManager = do
  status <- primIO prim__validateGPUMachineManager
  status <- onCollectAny status Status.delete
  pure (MkStatus status)

export
gpuMachineManager : IO Platform
gpuMachineManager = do
  platform <- primIO prim__gpuMachineManager
  pure (MkPlatform platform)
