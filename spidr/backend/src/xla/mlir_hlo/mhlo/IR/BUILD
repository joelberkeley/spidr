cc_library(
    name = "IR",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@xla//xla/mlir_hlo:hlo_dialect_registration",
        "//src/mlir/IR",
    ],
    visibility = ["//visibility:public"],
)
