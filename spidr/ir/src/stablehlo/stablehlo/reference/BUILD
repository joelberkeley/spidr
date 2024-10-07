cc_library(
    name = "reference",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "//src/mlir/IR",
        "//src/mlir/Tools/PDLL/AST",
        "@stablehlo//:reference_ops",
    ],
    visibility = ["//visibility:public"],
)
