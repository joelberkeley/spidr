/*
Copyright (C) 2024  Joel Berkeley

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/
#include <string>

#include "xla/pjrt/pjrt_executable.h"

#include "../client/executable_build_options.h"
#include "../../ffi.h"

extern "C" {
    struct CompileOptions;

    CompileOptions* CompileOptions_new(ExecutableBuildOptions* executable_build_options) {
        auto executable_build_options_ = reinterpret_cast<xla::ExecutableBuildOptions*>(
            executable_build_options
        );
        auto options = new xla::CompileOptions{
            .argument_layouts = std::nullopt,
            .executable_build_options = *executable_build_options_,
            .env_option_overrides = {},
            .target_config = std::nullopt,
        };
        return reinterpret_cast<CompileOptions*>(options);
    }

    string* CompileOptions_SerializeAsString(CompileOptions* s) {
        auto s_ = reinterpret_cast<xla::CompileOptions*>(s);
        auto res = s_->ToProto()->SerializeAsString();
        return reinterpret_cast<string*>(new std::string(res));
    }
}
