cc_library(
    name = "dialect",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@llvm-project//mlir:IR",
        "@stablehlo//:register",
        "@stablehlo//:stablehlo_serialization",
        "//src/mlir/IR",
    ],
    visibility = ["//visibility:public"],
)
