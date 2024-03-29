cc_library(
    name = "service",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]) + [
        "@xla//xla/service:platform_util",
    ],
    hdrs = glob(["*.h"]),
    visibility = ["//visibility:public"],
)
