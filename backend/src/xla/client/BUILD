cc_library(
    name = "client",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@xla//xla/client:client",
        "@xla//xla/client:client_library",
        "//src:src",
        "//src/xla:xla",
        "//src/xla/stream_executor:stream_executor",
    ],
    visibility = ["//visibility:public"],
)
