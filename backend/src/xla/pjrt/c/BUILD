cc_library(
    name = "pjrt_c",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.h"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@xla//xla/pjrt:pjrt_client",
    ],
    visibility = ["//visibility:public"],
)
