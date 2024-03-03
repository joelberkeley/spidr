cc_library(
    name = "pjrt_c",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.c"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@xla//xla/pjrt:pjrt_client",  // not sure about this
    ],
    visibility = ["//visibility:public"],
)
