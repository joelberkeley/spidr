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
// we have included this as it appears to be the source of HloModuleProto, but
// can't find it, so we'll rely on a transitive BUILD target
#include "../service/hlo.pb.h"

extern "C" {
    struct XlaComputation;

    void XlaComputation_delete(XlaComputation* s);
    const HloModuleProto& XlaComputation_proto(XlaComputation* s);
}
