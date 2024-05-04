cc_library(
    name = "client",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@xla_extension//:xla_extension",
        "//src:src",
        "//src/tensorflow/compiler/xla:xla",
        "//src/tensorflow/stream_executor:stream_executor",
    ],
    visibility = ["//visibility:public"],
)
