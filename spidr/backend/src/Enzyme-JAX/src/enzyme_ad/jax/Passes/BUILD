cc_library(
    name = "Passes",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@xla//xla/hlo/builder:xla_builder",
        "@xla//xla/hlo/translate:stablehlo",
        "@xla//xla/hlo/builder/lib:math",
        "@xla//xla/mlir_hlo:hlo_dialect_registration",
        "@enzyme-jax//:everything",
        "//src/mlir/IR",
        "//src/mlir/Pass",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "example",
    linkstatic = True,
    srcs = glob(["*.cpp"]),
    deps = [
        "@xla//xla/hlo/builder:xla_builder",
        "@xla//xla/hlo/translate:stablehlo",
        "@xla//xla/hlo/builder/lib:math",
        "@xla//xla/mlir_hlo:hlo_dialect_registration",
        "@enzyme-jax//:everything",
        "//src/mlir/IR",
        "//src/mlir/Pass",
    ],
    visibility = ["//visibility:public"],
)
