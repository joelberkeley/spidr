/*
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
*/
#include <string_view>

#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"

#include "platform.h"
#include "platform_manager.h"

extern "C" {
    Status* PlatformManager_RegisterPlatform(Platform* platform) {
        auto platform_ = reinterpret_cast<stream_executor::Platform*>(platform);
        auto status = stream_executor::PlatformManager::RegisterPlatform(
            std::unique_ptr<stream_executor::Platform>(platform_)
        );
        return reinterpret_cast<Status*>(new tsl::Status(status));
    }

    Platform* PlatformManager_PlatformWithName(char* target) {
        std::string_view target_ = target;
        auto platform = *stream_executor::PlatformManager::PlatformWithName(target_);
        return reinterpret_cast<Platform*>(platform);
    }
}
