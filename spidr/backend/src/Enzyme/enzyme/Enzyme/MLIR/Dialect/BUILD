cc_library(
    name = "Dialect",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@stablehlo//:register",
        "@enzyme//:EnzymeMLIR",
        "@llvm-project//mlir:IR",
        "//src/mlir/IR",
    ],
    visibility = ["//visibility:public"],
)
