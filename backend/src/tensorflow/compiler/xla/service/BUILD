cc_library(
    name = "service",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@xla_extension//:xla_extension",
        "//src/tensorflow/stream_executor:stream_executor",
    ],
    visibility = ["//visibility:public"],
)
