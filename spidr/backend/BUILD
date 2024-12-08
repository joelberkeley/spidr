# while it seems counterintuitive to build a shared library with a cc_binary rule when
# there's a cc_library rule, the docs explain how cc_binary with linkshared = True
# produce a shared library:
#
# > it is assumed that shared libraries built with a cc_binary rule are only
# > loaded manually by other programs
#
# see also this SO post https://stackoverflow.com/a/51689102/5986907

cc_binary(
    name = "c_xla",
    linkshared = True,
    linkstatic = True,
    srcs = [
        "//src/mlir/IR",
        "//src/stablehlo/dialect",
        "//src/xla",
        "//src/xla/client",
        "//src/xla/hlo/builder",
        "//src/xla/hlo/builder/lib",
        "//src/xla/hlo/translate",
        "//src/xla/mlir_hlo/mhlo/IR",
        "//src/xla/pjrt",
        "//src/xla/pjrt/c",
        "//src/xla/service",
        "//src",
    ],
    deps = [
        "//src/mlir/IR",
        "//src/stablehlo/dialect",
        "//src/xla",
        "//src/xla/client",
        "//src/xla/hlo/builder",
        "//src/xla/hlo/builder/lib",
        "//src/xla/hlo/translate",
        "//src/xla/mlir_hlo/mhlo/IR",
        "//src/xla/pjrt",
        "//src/xla/pjrt/c",
        "//src/xla/service",
        "//src",
    ],
)
