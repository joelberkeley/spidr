cc_library(
    name = "pjrt",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.c"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@xla//xla/pjrt:pjrt_executable",
    ],
    visibility = ["//visibility:public"],
)
