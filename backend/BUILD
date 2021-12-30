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

cc_import(
    name = "xla_extension_internal",
    hdrs = glob(["xla_extension/include/**"]),
    shared_library = "xla_extension/lib/libxla_extension.so",
)

# Note this cc_library indirection is a workaround to get the include statements working
cc_library(
    name = "xla_extension",
    includes = ["xla_extension/include"],
    deps = [":xla_extension_internal"],
)

cc_library(
    name = "xla",
    srcs = glob(["src/**/*.cpp"]),
    linkopts = [
        "-L/home/joel/git/joelberkeley/spidr/backend/xla_extension/lib",
        "-lxla_extension",
        "-Wl,-rpath=backend/xla_extension/lib/",
    ],
    deps = [":xla_extension"],
)
