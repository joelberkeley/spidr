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
#include "absl/types/span.h"
#include "xla/client/local_client.h"

#include "../literal.h"
#include "local_client.h"

extern "C" {
    GlobalData* LocalClient_TransferToServer(LocalClient& client, Literal& literal) {
        xla::LocalClient& client_ = reinterpret_cast<xla::LocalClient&>(client);
        xla::Literal& literal_ = reinterpret_cast<xla::Literal&>(literal);

        std::unique_ptr<xla::GlobalData> global_data = client_.TransferToServer(literal_).value();

        return reinterpret_cast<GlobalData*>(global_data.release());
    }

    Literal* LocalClient_ExecuteAndTransfer(
        LocalClient& client,
        XlaComputation& computation,
        GlobalData** arguments,
        int arguments_len
    ) {
        xla::LocalClient& client_ = reinterpret_cast<xla::LocalClient&>(client);
        xla::XlaComputation& computation_ = reinterpret_cast<xla::XlaComputation&>(computation);
        xla::GlobalData** arguments_ = reinterpret_cast<xla::GlobalData**>(arguments);

        auto arguments_span = absl::Span<xla::GlobalData* const>(arguments_, arguments_len);
        auto lit = client_.ExecuteAndTransfer(computation_, arguments_span).value();

        xla::Literal* res = new xla::Literal(lit.shape(), false);
        res->MoveFrom(std::move(lit));

        return reinterpret_cast<Literal*>(res);
    }
}
