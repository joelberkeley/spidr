/*
Copyright 2025 Joel Berkeley

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
#include "llvm/Support/raw_ostream.h"

#include "../../ffi.h"
#include "raw_ostream.h"

extern "C" {
    struct raw_string_ostream;

    raw_string_ostream* raw_string_ostream_new(string& o) {
        auto& o_ = reinterpret_cast<std::string&>(o);
        return reinterpret_cast<raw_string_ostream*>(new llvm::raw_string_ostream(o_));
    }

    void raw_string_ostream_delete(raw_string_ostream* s) {
        delete reinterpret_cast<llvm::raw_string_ostream*>(s);
    }
}
