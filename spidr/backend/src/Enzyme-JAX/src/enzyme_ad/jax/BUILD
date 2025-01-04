cc_library(
    name = "jax",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@enzyme-jax//:everything",
        "//src/mlir/IR",
    ],
    visibility = ["//visibility:public"],
)
