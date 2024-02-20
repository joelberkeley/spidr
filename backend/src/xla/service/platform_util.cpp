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
#include "xla/service/platform_util.h"

#include "../stream_executor/platform.h"

extern "C" {
    Platform* PlatformUtil_GetPlatform(const char* platform_name) {
        std::cout << "PlatformUtil_GetPlatform ..." << std::endl;
        auto* platform = xla::PlatformUtil::GetPlatform(platform_name).value();
        std::cout << "... successfully called GetPlatform() on " << platform_name << std::endl;
        return reinterpret_cast<Platform*>(platform);
    }
}
