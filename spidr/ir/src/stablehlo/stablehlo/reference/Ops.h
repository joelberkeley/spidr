/*
Copyright 2024 Joel Berkeley

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
#include <stdint.h>

#include "../../../mlir/IR/Attributes.h"
#include "../../../mlir/IR/Region.h"
#include "../../../mlir/IR/StandardTypes.h"
#include "../../../mlir/Tools/PDLL/AST/Types.h"

#include "Axes.h"
#include "Configuration.h"
#include "Index.h"
#include "Process.h"
#include "Scope.h"
#include "Tensor.h"
#include "Value.h"

extern "C" {
    // where does this live? I can't find its definition - seems like an enum
    struct ComparisonDirection;

    Tensor* absOp(Tensor& operand, ShapedType& resultType);
    Tensor* addOp(Tensor& lhs, Tensor& rhs, ShapedType& resultType);
    Tensor* andOp(Tensor& lhs, Tensor& rhs, ShapedType& resultType);
    Tensor* atan2Op(Tensor& lhs, Tensor& rhs, ShapedType& resultType);
    Tensor* bitcastConvertOp(Tensor& operand, ShapedType& resultType);
    Tensor* broadcastInDimOp(Tensor& operand, Axes& broadcastDimensions, ShapedType& resultType);
    Tensor* cbrtOp(Tensor& operand, ShapedType& resultType);
    Tensor* ceilOp(Tensor& operand, ShapedType& resultType);
    Tensor* clampOp(Tensor& min, Tensor& operand, Tensor& max, ShapedType& resultType);
    Tensor* clzOp(Tensor& operand, ShapedType& resultType);
    Tensor* compareOp(
        Tensor& lhs, Tensor& rhs, ComparisonDirection& comparisonDirection, ShapedType& resultType
    );
    Tensor* concatenateOp(Tensor* inputs, int64_t dimension, ShapedType& resultType);
    Tensor* constantOp(ElementsAttr& value);
    Tensor* convertOp(Tensor& operand, ShapedType& resultType);
    Tensor* cosineOp(Tensor& operand, ShapedType& resultType);
    Tensor* divideOp(Tensor& lhs, Tensor& rhs, ShapedType& resultType);
    Tensor* dotGeneralOp(
        Tensor& lhs,
        Tensor& rhs,
        Axes& lhsBatchingDimensions,
        Axes& rhsBatchingDimensions,
        Axes& lhsContractingDimensions,
        Axes& rhsContractingDimensions,
        ShapedType& resultType
    );
    Tensor* dynamicSliceOp(
        Tensor& operand, Tensor* startIndices, Sizes& sliceSizes, ShapedType& resultType
    );
    Tensor* dynamicUpdateSliceOp(
        Tensor& operand, Tensor& update, Tensor* startIndices, ShapedType& resultType
    );
    Tensor* exponentialOp(Tensor& operand, ShapedType& resultType);
    Tensor* floorOp(Tensor& operand, ShapedType& resultType);
    InterpreterValue* getTupleElementOp(Tuple& operand, int32_t index);
    InterpreterValue* ifOp(
        Tensor& pred, Region& trueBranch, Region& falseBranch, Process* process, Scope& scope
    );
    Tensor* iotaOp(int64_t iotaDimension, ShapedType& resultType);
    Tensor* isFiniteOp(Tensor& operand, ShapedType& resultType);
    Tensor* log1pOp(Tensor& operand, ShapedType& resultType);
    Tensor* logOp(Tensor& operand, ShapedType& resultType);
    Tensor* logisticOp(Tensor& operand, ShapedType& resultType);
    Tensor* mapOp(
        Tensor* inputs,
        Region& computation,
        Process* process,
        Scope& scope,
        ShapedType& resultType
    );
    Tensor* maxOp(Tensor& lhs, Tensor& rhs, ShapedType& resultType);
    Tensor* minOp(Tensor& lhs, Tensor& rhs, ShapedType& resultType);
    Tensor* multiplyOp(Tensor& lhs, Tensor& rhs, ShapedType& resultType);
    Tensor* negOp(Tensor& operand, ShapedType& resultType);
    Tensor* notOp(Tensor& operand, ShapedType& resultType);
    Tensor* orOp(Tensor& lhs, Tensor& rhs, ShapedType& resultType);
    Tensor* powerOp(Tensor& lhs, Tensor& rhs, ShapedType& resultType);
    Tensor* reduceOp(
        Tensor* inputs,
        Tensor* initValues,
        Axes& dimensions,
        Region& body,
        Process *process, Scope& scope,
        ShapedType* resultTypes
    );
    Tensor* reducePrecisionOp(
        Tensor& operand, int32_t exponentBits, int32_t mantissaBits, ShapedType& resultType
    );
    Tensor* remOp(Tensor& lhs, Tensor& rhs, ShapedType& resultType);
    Tensor* reshapeOp(Tensor& operand, ShapedType& resultType);
    Tensor* reverseOp(Tensor& operand, Axes& dimensions, ShapedType& resultType);
    Tensor* roundOp(Tensor& operand, ShapedType& resultType);
    Tensor* roundNearestEvenOp(Tensor& operand, ShapedType& resultType);
    Tensor* rsqrtOp(Tensor& operand, ShapedType& resultType);
    Tensor* selectOp(Tensor& pred, Tensor& onTrue, Tensor& onFalse, ShapedType& resultType);
    Tensor* signOp(Tensor& operand, ShapedType& resultType);
    Tensor* sineOp(Tensor& operand, ShapedType& resultType);
    Tensor* sliceOp(Tensor& operand, Sizes& startIndices, Sizes& strides, ShapedType& resultType);
    Tensor* sortOp(
        Tensor* inputs,
        int64_t dimension,
        bool isStable,
        Region& comparator,
        Process* process,
        Scope& scope
    );
    Tensor* sqrtOp(Tensor& operand, ShapedType& resultType);
    Tensor* subtractOp(Tensor& lhs, Tensor& rhs, ShapedType& resultType);
    Tensor* tanhOp(Tensor& operand, ShapedType& resultType);
    Tensor* transposeOp(Tensor& operand, Axes& permutation, ShapedType& resultType);
    Tuple* tupleOp(InterpreterValue* val, TupleType& resultType);
    Tensor* xorOp(Tensor& lhs, Tensor& rhs, ShapedType& resultType);

    InterpreterValue* eval(
        Region& region,
        InterpreterValue* args,
        InterpreterFallback* fallback,
        Process* process,
        Scope* parent
    );
}
