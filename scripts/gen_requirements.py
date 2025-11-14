import argparse
import re
import tomli


def read_deps(root_config, outer_name, inner_name):
    outer_config = root_config.get(outer_name, {})
    return outer_config.get(inner_name, [])


def parse_reqs(select_min, reqs):
    selected_reqs = []
    for req in reqs:
        req = req.strip()
        # Extract package name and version range
        match = re.match(r"([^>=<\s]+)\s*>=\s*([^,]+),\s*<=\s*([^,]+)", req)
        if match:
            package, min_ver, max_ver = match.groups()
            selected_ver = min_ver if select_min else max_ver
            selected_reqs.append(f"{package}=={selected_ver}")
        else:
            raise Exception(f"ERROR: Unmatched dependency found: {req}")
    return selected_reqs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate requirements.txt")
    parser.add_argument(
        "version_mode",
        choices=["min", "max", "passthrough"],
        help="Which version to select: 'min', 'max', or 'passthrough'.",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Where to write the output (e.g., requirements.txt).",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Use the dev dependency group.",
    )
    args = parser.parse_args()

    with open("pyproject.toml", "rb") as f:
        config = tomli.load(f)

    if args.dev:
        deps = read_deps(config, "dependency-groups", "dev")
    else:
        deps = read_deps(config, "project", "dependencies")

    if args.version_mode != "passthrough":
        deps = parse_reqs(args.version_mode == "min", deps)

    with open(args.output_file, "w") as f:
        f.write("\n".join(deps))
