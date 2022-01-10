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
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"

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
        Literal** literals,
        int literals_len
    ) {
        xla::LocalClient& client_ = reinterpret_cast<xla::LocalClient&>(client);
        xla::XlaComputation& computation_ = reinterpret_cast<xla::XlaComputation&>(computation);
        xla::Literal** literals_ = reinterpret_cast<xla::Literal**>(literals);

        xla::GlobalData* arguments[literals_len];
        for (int i = 0; i < literals_len; i++) {
            arguments[i] = client_.TransferToServer(*(literals_[i])).ConsumeValueOrDie().release();
        }

        auto arguments_span = absl::Span<xla::GlobalData* const>(arguments, literals_len);
        xla::Literal lit =
            client_.ExecuteAndTransfer(computation_, arguments_span).ConsumeValueOrDie();

        xla::Literal* res = new xla::Literal(lit.shape(), true);
        *res = lit.Clone();
        return reinterpret_cast<Literal*>(res);
    }
}

xla::GlobalData* sample_harness1(xla::LocalClient* client, xla::Literal& lit) {
    auto global_data = client->TransferToServer(lit).ConsumeValueOrDie();

    xla::GlobalData* global_data_non_stack =
        new xla::GlobalData(client->stub(), global_data->handle());

    std::vector<std::unique_ptr<xla::GlobalData>> to_release;
    to_release.push_back(std::move(global_data));
    xla::GlobalData::Release(std::move(to_release));
    return global_data_non_stack;
}

extern "C" {
    void sample_harness() {
        xla::LocalClient* client(xla::ClientLibrary::LocalClientOrDie());

        // Transfer parameters.
        xla::Literal param0_literal =
            xla::LiteralUtil::CreateR2<float>({{1.1f, 2.2f, 3.3f, 5.5f}});
        xla::GlobalData* param0_data =
            sample_harness1(client, param0_literal);

        xla::Literal param1_literal = xla::LiteralUtil::CreateR2<float>(
            {{3.1f, 4.2f, 7.3f, 9.5f}, {1.1f, 2.2f, 3.3f, 4.4f}});
        xla::GlobalData* param1_data =
            sample_harness1(client, param1_literal);

        // Build computation.
        xla::XlaBuilder builder("");
        auto p0 = Parameter(&builder, 0, param0_literal.shape(), "param0");
        auto p1 = Parameter(&builder, 1, param1_literal.shape(), "param1");
        Add(p0, p1);

        xla::StatusOr<xla::XlaComputation> computation_status = builder.Build();
        xla::XlaComputation computation = computation_status.ConsumeValueOrDie();

        // Execute and transfer result of computation.
        xla::ExecutionProfile profile;
        xla::StatusOr<xla::Literal> result = client->ExecuteAndTransfer(
            computation,
            /*arguments=*/{param0_data, param1_data},
            /*execution_options=*/nullptr,
            /*execution_profile=*/&profile);
        xla::Literal actual = result.ConsumeValueOrDie();

        LOG(INFO) << absl::StrFormat("computation took %dns",
                                    profile.compute_time_ns());
        LOG(INFO) << actual.ToString();
    }
}
