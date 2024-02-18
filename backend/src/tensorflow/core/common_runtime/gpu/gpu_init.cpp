/*
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
*/
#include "xla/stream_executor/gpu/gpu_init.h"

#include "../../platform/status.h"
#include "../../../stream_executor/platform.h"

extern "C" {
    Status* ValidateGPUMachineManager() {
        tensorflow::Status status = tensorflow::ValidateGPUMachineManager();
        return reinterpret_cast<Status*>(new tensorflow::Status(status));
    }

    Platform* GPUMachineManager() {
        return reinterpret_cast<Platform*>(tensorflow::GPUMachineManager());
    }
}
