cc_library(
    name = "translate",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@xla//xla/hlo/translate:stablehlo",
        "//src/mlir",
        "//src/xla/service",
    ],
    visibility = ["//visibility:public"],
)
