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
||| A tensor graph is executed in a physical device. This module provides handles for devices.
module Device

import Control.Monad.Maybe

import Compiler.TensorFlow.Compiler.Xla.Service.PlatformUtil
import Compiler.TensorFlow.Core.CommonRuntime.GPU.GPUInit
import Compiler.TensorFlow.Core.Platform.Status
import Compiler.TensorFlow.StreamExecutor.Platform

||| A handle to a physical device, used to execute a computational graph.
public export
data Device = MkDevice Platform

||| A handle to a CUDA-enabled GPU, if such a device is found.
export
cuda : MaybeT IO Device
cuda = MkDevice <$> if (ok !validateGPUMachineManager) then gpuMachineManager else nothing

||| A handle to the host CPU.
export
cpu : IO Device
cpu = MkDevice <$> getPlatform "Host"
