cc_library(
    name = "dialect",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@stablehlo//:register",
        "//src/mlir/IR",
    ],
    visibility = ["//visibility:public"],
)
