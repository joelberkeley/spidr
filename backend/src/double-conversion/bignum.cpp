/*
Copyright 2021 Joel Berkeley

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
/* This file contains the pure C API to XLA. */
#include <double-conversion/bignum.h>

extern "C"
{
    struct cBignum;

    using namespace double_conversion;

    struct cBignum* cBignum_new(uint64_t x) {
        Bignum* bn = new Bignum();
        bn->AssignUInt64(x);
        return reinterpret_cast<cBignum*>(bn);
    }

    void cBignum_del(struct cBignum* s) {
        delete reinterpret_cast<Bignum*>(s);
    }

    struct cBignum* cBignum_add(struct cBignum& s, struct cBignum& other) {
        Bignum* sum;
        sum->AddBignum(reinterpret_cast<Bignum&>(s));
        sum->AddBignum(reinterpret_cast<Bignum&>(other));
        // todo is this a dangling pointer?
        return reinterpret_cast<cBignum*>(sum);
    }

    int cBignum_compare(struct cBignum& s, struct cBignum& other) {
        return Bignum::Compare(
            reinterpret_cast<Bignum&>(s),
            reinterpret_cast<Bignum&>(other)
        );
    }
}
