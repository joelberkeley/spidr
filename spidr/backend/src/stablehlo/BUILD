cc_library(
    name = "stablehlo",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["**/*.cpp"]),
    hdrs = glob(["**/*.h"]),
    deps = [
        "@stablehlo//:register",
        "@stablehlo//:stablehlo_serialization",
        "//src/llvm",
        "//src/mlir",
    ],
    visibility = ["//visibility:public"],
)
