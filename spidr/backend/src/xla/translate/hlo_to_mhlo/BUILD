cc_library(
    name = "hlo_to_mhlo",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@xla//xla/translate:hlo_to_mlir_hlo",
        "//src/xla/service",
    ],
    visibility = ["//visibility:public"],
)
