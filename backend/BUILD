# Copyright 2021 Joel Berkeley
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cc_binary(
    name = "c_xla_extension",
    linkshared = True,
    srcs = [
        "//src/tensorflow/core/common_runtime/gpu:gpu",
        "//src/tensorflow/compiler/xla:xla",
        "//src/tensorflow/compiler/xla/client:client",
        "//src/tensorflow/compiler/xla/client/lib:lib",
        "//src/tensorflow/compiler/xla/service:service",
        "//src/tensorflow/stream_executor:stream_executor",
        "//src:src",
    ],
    deps = [
        "//src/tensorflow/core/common_runtime/gpu:gpu",
        "//src/tensorflow/compiler/xla:xla",
        "//src/tensorflow/compiler/xla/client:client",
        "//src/tensorflow/compiler/xla/client/lib:lib",
        "//src/tensorflow/compiler/xla/service:service",
        "//src/tensorflow/stream_executor:stream_executor",
        "//src:src",
    ],
)
