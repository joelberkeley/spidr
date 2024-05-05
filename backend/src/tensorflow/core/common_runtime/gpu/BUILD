cc_library(
    name = "gpu",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@xla_extension//:xla_extension",
        "//src/tensorflow/core/platform:platform",
        "//src/tensorflow/stream_executor:stream_executor",
    ],
    visibility = ["//visibility:public"],
)
