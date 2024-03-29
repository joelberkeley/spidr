cc_library(
    name = "pjrt_c",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]) + [
        "@xla//xla/pjrt/c:pjrt_c_api_cpu",
    ],
    hdrs = glob(["*.h"]),
    deps = [
        "//src/xla:xla",
        "//src/xla/pjrt:pjrt",
    ],
    visibility = ["//visibility:public"],
)
