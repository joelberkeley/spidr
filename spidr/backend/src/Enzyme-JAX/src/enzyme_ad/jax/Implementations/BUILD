cc_library(
    name = "Implementations",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@enzyme-jax//:enzymexlamlir-opt",
        "//src/mlir/IR",
    ],
    visibility = ["//visibility:public"],
)
