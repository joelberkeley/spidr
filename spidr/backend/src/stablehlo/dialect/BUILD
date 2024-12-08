cc_library(
    name = "dialect",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@stablehlo//:register",
        "@stablehlo//:stablehlo_serialization",
        "//src/mlir/IR",
    ],
    visibility = ["//visibility:public"],
)
