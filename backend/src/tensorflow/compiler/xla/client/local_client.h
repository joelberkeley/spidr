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
#include "../literal.h"
#include "global_data.h"
#include "xla_computation.h"

extern "C" {
    struct LocalClient;

    GlobalData* LocalClient_TransferToServer(LocalClient& client, Literal& literal);

    Literal* LocalClient_ExecuteAndTransfer(
        LocalClient& client,
        XlaComputation& computation,
        GlobalData** arguments,
        int arguments_len
    );
}
