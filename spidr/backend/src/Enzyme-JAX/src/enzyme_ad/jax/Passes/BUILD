# requires modifications to "@enzyme-jax//src/enzyme_ad/jax:XLADerivatives",
# add visibility = ["//visibility:public"], and remove (or fix)
# copts = [
#     "-Werror=unused-variable",
#     "-Werror=unused-but-set-variable",
#     "-Werror=return-type",
#     "-Werror=unused-result",
# ],

cc_library(
    name = "Passes",
    linkstatic = True,
    alwayslink = True,
    visibility = ["//visibility:public"],
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "//src/mlir/IR",
        "//src/mlir/Pass",

        "@llvm-project//mlir:AllPassesAndDialects",
        "@enzyme//:EnzymeMLIR",
        "@enzyme-jax//src/enzyme_ad/jax:XLADerivatives",
    ],
)

# cc_binary(
#     name = "example",
#     linkstatic = True,
#     srcs = glob(["*.cpp"]),
#     deps = [
#         "@xla//xla/hlo/builder:xla_builder",
#         "@xla//xla/hlo/translate:stablehlo",
#         "@xla//xla/hlo/builder/lib:math",
#         "@xla//xla/mlir_hlo:hlo_dialect_registration",
#         "@enzyme-jax//:everything",
#         "//src/mlir/IR",
#         "//src/mlir/Pass",
#     ],
#     visibility = ["//visibility:public"],
# )
