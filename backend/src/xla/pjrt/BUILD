cc_library(
    name = "pjrt",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]) + [
        "@xla//xla/client:xla_computation",
        "@xla//xla/pjrt:pjrt_executable",
    ],
    hdrs = glob(["*.h"]),
    deps = [
        "//src/xla/service:service",
    ],
    visibility = ["//visibility:public"],
)
