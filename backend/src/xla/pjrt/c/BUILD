cc_library(
    name = "pjrt_c",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@xla//xla/pjrt/c:pjrt_c_api_cpu",
        "//src/xla:xla",
        "//src/xla/pjrt:pjrt",
    ],
    visibility = ["//visibility:public"],
)
