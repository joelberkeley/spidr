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
||| This module contains functionality for accessing hardware devices on the current machine.
module Device

import Compiler.TensorFlow.Compiler.XLA.Service.PlatformUtil
import Compiler.TensorFlow.Core.CommonRuntime.GPU.GPUInit
import Compiler.TensorFlow.Core.Platform.Status

||| A handle to an installed computational device, such as a CPU or GPU.
export
data Device = CPU AnyPtr | GPU AnyPtr

export
platform : Device -> AnyPtr
platform (CPU ptr) = ptr
platform (GPU ptr) = ptr

||| A handle to an installed GPU, if present.
export
gpu : Maybe Device
-- unsafePerformIO is safe as long as it's the only way to get a `GPU`
gpu = unsafePerformIO $ do
  status <- primIO prim__validateGPUMachineManager
  status <- onCollectAny status Status.delete
  case prim__ok status of
    True => do
      ptr <- primIO prim__gpuMachineManager
      pure (Just (GPU ptr))
    False => pure Nothing

||| A handle to the host CPU.
export
cpu : Device
-- unsafePerformIO is safe as long as it's the only way to get a `CPU`
cpu = CPU $ unsafePerformIO $ primIO (prim__getPlatform "Host")
