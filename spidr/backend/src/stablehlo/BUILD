cc_library(
    name = "stablehlo",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["**/*.cpp"]),
    hdrs = glob(["**/*.h"]),
    deps = [
        "@stablehlo//:register",
        "//src/mlir",
    ],
    visibility = ["//visibility:public"],
)
