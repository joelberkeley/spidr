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

#include "../../status.h"
#include "../platform.h"

extern "C" {
    Status* ValidateGPUMachineManager() {
        std::cout << "ValidateGPUMachineManager ..." << std::endl;
        tsl::Status status = stream_executor::ValidateGPUMachineManager();
        std::cout << "... return" << std::endl;
        return reinterpret_cast<Status*>(new tsl::Status(status));
    }

    Platform* GPUMachineManager() {
        std::cout << "GPUMachineManager ..." << std::endl;
        std::cout << "... return" << std::endl;
        return reinterpret_cast<Platform*>(stream_executor::GPUMachineManager());
    }
}
