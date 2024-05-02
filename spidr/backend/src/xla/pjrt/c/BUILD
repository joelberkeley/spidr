cc_library(
    name = "c",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        "@xla//xla/pjrt/c:pjrt_c_api_helpers",
    ],
    visibility = ["//visibility:public"],
)
