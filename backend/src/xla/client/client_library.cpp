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
#include "xla/client/client_library.h"
#include "absl/types/optional.h"
#include "xla/stream_executor/host/host_platform_id.h"
#include "xla/client/local_client.h"
#include "xla/service/platform_util.h"

#include "../stream_executor/platform.h"
#include "client_library.h"
#include <cstddef>
#include <set>

extern "C" {
    LocalClient* ClientLibrary_GetOrCreateLocalClient(
        Platform* platform, int* allowed_devices, int allowed_devices_len
    ) {
        auto platform_ = reinterpret_cast<xla::se::Platform*>(platform);
        auto client = xla::ClientLibrary::GetOrCreateLocalClient(platform_, std::nullopt).value();

        return reinterpret_cast<LocalClient*>(client);
    }
}
