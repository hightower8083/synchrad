"""
Prints relevant information regarding the capabilities of the current OpenCL runtime and devices
Note that pyopencl has a script that prints all properties in its examples folder
"""

import pyopencl as cl

import math


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


print("PyOpenCL version: " + cl.VERSION_TEXT)
print("OpenCL header version: " + ".".join(map(str, cl.get_cl_header_version())) + "\n")

# Get installed platforms (SDKs)
print("- Installed platforms (SDKs) and available devices:")
platforms = cl.get_platforms()

for plat in platforms:
    indent = ""

    # Get and print platform info
    print(indent + "{} ({})".format(plat.name, plat.vendor))
    indent = "\t"
    print(indent + "Version: " + plat.version)
    print(indent + "Profile: " + plat.profile)

    # Get and print device info
    devices = plat.get_devices(cl.device_type.ALL)

    print(indent + "Available devices: ")
    if not devices:
        print(indent + "\tNone")

    for dev in devices:
        indent = "\t\t"
        print(indent + "{} ({})".format(dev.name, dev.vendor))

        indent = "\t\t\t"
        flags = [
            ("Version", dev.version),
            ("Type", cl.device_type.to_string(dev.type)),
            ("Memory (global)", convert_size(dev.global_mem_size)),
            ("Memory (local)", convert_size(dev.local_mem_size)),
            ("Device available", str(bool(dev.available))),
            ("Compiler available", str(bool(dev.compiler_available))),
        ]

        [
            print(indent + "{0:<25}{1:<10}".format(name + ":", flag))
            for name, flag in flags
        ]

        # Device version string has the following syntax, extract the number like this
        # OpenCL<space><major_version.minor_version><space><vendor-specific information>
        version_number = float(dev.version.split(" ")[1])

    print("")
