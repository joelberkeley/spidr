cc_library(
    name = "pjrt_c",
    linkstatic = True,
    alwayslink = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.h"]),
    deps = [
        # do i need to move _just_ the XLA dependencies to sources, now that they're
        # not in a companion .so file?
        "@xla//xla/pjrt/c:pjrt_c_api_cpu",
        "//src/xla:xla",
        "//src/xla/pjrt:pjrt",
    ],
    visibility = ["//visibility:public"],
)
