cc_library(
    name = "stream_executor",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@xla//xla/stream_executor:platform_manager",
    ],
    visibility = ["//visibility:public"],
)
