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
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/local_client.h"

#include "client_library.h"

extern "C" {
    struct ClientLibrary;

    LocalClient* ClientLibrary_LocalClientOrDie() {
        xla::LocalClient* client = xla::ClientLibrary::LocalClientOrDie();
        return reinterpret_cast<LocalClient*>(client);
    }
}
