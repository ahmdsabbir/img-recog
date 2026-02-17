import shlex


def parse_command(command_str: str) -> dict | None:
    """
    Parse an interactive command string into a flat argument dict.
    Returns None if the input is empty.
    """
    parts = shlex.split(command_str)
    if not parts:
        return None

    cmd_args = {
        "command": parts[0],
        "image": None,
        "products_dir": None,
        "save_preprocessed": False,
        "preprocessed_dir": None,
        "category": None,
        "attribute": None,
        "use_trained": False,
        "clear": False,
        "list": False,
    }

    i = 1
    while i < len(parts):
        p = parts[i]

        if p == "--image" and i + 1 < len(parts):
            cmd_args["image"] = parts[i + 1]
            i += 2
        elif p == "--products_dir" and i + 1 < len(parts):
            cmd_args["products_dir"] = parts[i + 1]
            i += 2
        elif p == "--preprocessed_dir" and i + 1 < len(parts):
            cmd_args["preprocessed_dir"] = parts[i + 1]
            i += 2
        elif p == "--category" and i + 1 < len(parts):
            cmd_args["category"] = parts[i + 1]
            i += 2
        elif p == "--attribute" and i + 1 < len(parts):
            cmd_args["attribute"] = parts[i + 1]
            i += 2
        elif p == "--save-preprocessed":
            cmd_args["save_preprocessed"] = True
            i += 1
        elif p == "--use-trained":
            cmd_args["use_trained"] = True
            i += 1
        elif p == "--clear":
            cmd_args["clear"] = True
            i += 1
        elif cmd_args["command"] == "cache":
            if p in ("list", "clear"):
                cmd_args[p] = True
            else:
                print(f"Unknown cache subcommand: {p}")
            i += 1
        else:
            print(f"Unknown argument: {p}")
            i += 1

    return cmd_args
