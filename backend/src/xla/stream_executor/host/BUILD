cc_library(
    name = "host",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@xla//xla/stream_executor/host:host_platform",
        "//src/xla:xla",
        "//src/xla/stream_executor:stream_executor",
    ],
    visibility = ["//visibility:public"],
)
