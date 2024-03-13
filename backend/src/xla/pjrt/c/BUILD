cc_library(
    name = "pjrt_c",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt/c:pjrt_c_api_cpu",
        "@xla//xla/pjrt/c:pjrt_c_api_helpers",
        "//src/xla:xla",
        "//src/xla/pjrt:pjrt",
    ],
    visibility = ["//visibility:public"],
)
