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
#include "tensorflow/compiler/xla/client/local_client.h"

#include "../literal.h"
#include "local_client.h"

extern "C" {
    GlobalData* LocalClient_TransferToServer(LocalClient& client, Literal& literal) {
        std::cout << "LocalClient_TransferToServer ... " << std::endl;
        xla::LocalClient& client_ = reinterpret_cast<xla::LocalClient&>(client);
        xla::Literal& literal_ = reinterpret_cast<xla::Literal&>(literal);

        std::unique_ptr<xla::GlobalData> global_data =
            client_.TransferToServer(literal_).ConsumeValueOrDie();

        std::cout << "LocalClient_TransferToServer ... copy GlobalData to heap" << std::endl;
        std::cout << "handle " << global_data.get()->handle().handle() << std::endl;

        xla::GlobalData* global_data_non_stack =
            new xla::GlobalData(client_.stub(), global_data->handle());

        std::vector<std::unique_ptr<xla::GlobalData>> to_release;
        to_release.push_back(std::move(global_data));
        xla::GlobalData::Release(std::move(to_release));

        return reinterpret_cast<GlobalData*>(global_data_non_stack);
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
        xla::Literal lit = client_.ExecuteAndTransfer(computation_, arguments_span).ConsumeValueOrDie();

        xla::Literal* res = new xla::Literal(lit.shape(), true);
        *res = lit.Clone();
        return reinterpret_cast<Literal*>(res);
    }

    Literal* LocalClient_ExecuteAndTransfer_parameter(
        LocalClient& client,
        XlaComputation& computation,
        Literal& literal0
        // Literal& literal1
    ) {
        xla::LocalClient& client_ = reinterpret_cast<xla::LocalClient&>(client);
        xla::XlaComputation& computation_ = reinterpret_cast<xla::XlaComputation&>(computation);
        xla::Literal& lit0 = reinterpret_cast<xla::Literal&>(literal0);
        // xla::Literal& lit1 = reinterpret_cast<xla::Literal&>(literal1);

        std::cout << lit0.ToString() << std::endl;
        // std::cout << lit1.ToString() << std::endl;

        xla::GlobalData* gd0 = client_.TransferToServer(lit0).ConsumeValueOrDie().get();
        // xla::GlobalData* gd1 = client_.TransferToServer(lit1).ConsumeValueOrDie().get();

        std::cout << gd0->handle().handle() << std::endl;
        // std::cout << gd1->handle().handle() << std::endl;

        xla::GlobalData* global_data[1];

        global_data[0] = gd0;
        // global_data[1] = gd1;

        auto arguments_span = absl::Span<xla::GlobalData* const>(global_data, 1);
        xla::Literal lit = client_.ExecuteAndTransfer(computation_, arguments_span).ConsumeValueOrDie();

        xla::Literal* res = new xla::Literal(lit.shape(), true);
        *res = lit.Clone();
        return reinterpret_cast<Literal*>(res);
    }
}
