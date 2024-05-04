cc_library(
    name = "lib",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@xla_extension//:xla_extension",
        "//src/tensorflow/compiler/xla:xla",
        "//src/tensorflow/compiler/xla/client:client",
    ],
    visibility = ["//visibility:public"],
)
