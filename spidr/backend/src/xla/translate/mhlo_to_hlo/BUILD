cc_library(
    name = "mhlo_to_hlo",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@xla//xla/translate:mlir_hlo_to_hlo",
        "//src/xla/service",
    ],
    visibility = ["//visibility:public"],
)
