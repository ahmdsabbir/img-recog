import os
import argparse
import shlex

from app.config import settings
from cli.container import Container
from cli.commands import rebuild, train, query, classify, cache
from cli.message import Message

Msg = Message()

def parse_command(command_str):
    """
    Parse interactive command string into argparse-like namespace.
    """
    parts = shlex.split(command_str)
    cmd_args = {
        "command": None,
        "image": None,
        "products_dir": None,
        "save_preprocessed": False,
        "preprocessed_dir": None,
        "category": None,
        "attribute": None,
        "use_trained": False,
        "cache_action": None,  # list, clear, delete, info
        "cache_key": None,  # for delete command
    }

    if not parts:
        return None

    cmd_args["command"] = parts[0]

    i = 1
    while i < len(parts):
        p = parts[i]

        # Flags with values
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
        elif p == "--key" and i + 1 < len(parts):
            cmd_args["cache_key"] = parts[i + 1]
            i += 2

        # Flags without values
        elif p == "--save-preprocessed":
            cmd_args["save_preprocessed"] = True
            i += 1
        elif p == "--use-trained":
            cmd_args["use_trained"] = True
            i += 1
        else:
            # Check if it's a cache subcommand
            if cmd_args["command"] == "cache":
                if p in ("list", "clear", "delete", "info"):
                    cmd_args["cache_action"] = p
                    i += 1
                else:
                    print(Msg.alert(f"Unknown cache subcommand: {p}"))
                    print(Msg.info("Valid subcommands: list, clear, delete, info"))
                    i += 1
            else:
                print(Msg.alert(f"Unknown argument: {p}"))
                i += 1

    return cmd_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["serve", "rebuild", "train"])
    parser.add_argument(
        "--products_dir",
        default="data/products",
        help="Directory of product images for rebuild",
    )
    parser.add_argument("--category", help="Category for training")
    parser.add_argument("--attribute", help="Attribute for training")
    args = parser.parse_args()

    container = Container()

    # ---------- Non-interactive rebuild ----------
    if args.command == "rebuild":
        rebuild.run_rebuild(
            container.embedding, container.vectore_store, args.products_dir
        )
        return

    # ---------- Non-interactive train ----------
    if args.command == "train":
        if not args.category or not args.attribute:
            print(Msg.info("Please provide --category and --attribute for training"))
            return
        train.run_train(args.category, args.attribute)
        return

    # ---------- Interactive serve ----------
    if args.command == "serve":
        print(Msg.highlight("Entering interactive serve mode. Type 'exit' to quit."))
        while True:
            try:
                command_str = input(Msg.highlight('\n>>> ')).strip()
                if command_str.lower() in ["exit", "quit"]:
                    print(Msg.info("Exiting serve..."))
                    break

                cmd = parse_command(command_str)
                if not cmd:
                    continue

                # ---------- REBUILD ----------
                if cmd["command"] == "rebuild":
                    products_dir = cmd["products_dir"] or "data/products"
                    rebuild.run_rebuild(
                        container.embedding, container.vectore_store, products_dir
                    )

                # ---------- QUERY ----------
                elif cmd["command"] == "query":
                    img_path = cmd["image"]
                    if not img_path or not os.path.exists(img_path):
                        print(Msg.alert("Please provide valid --image for query"))
                        continue

                    if not os.path.exists(settings.FAISS_INDEX_PATH):
                        print(Msg.alert("FAISS index not found. Rebuild index first."))
                        continue

                    query.run_query(
                        container.recommender, container.vectore_store, img_path
                    )

                # ---------- CLASSIFY ----------
                elif cmd["command"] == "classify":
                    img_path = cmd["image"]
                    use_trained = cmd["use_trained"]

                    if not img_path or not os.path.exists(img_path):
                        print("Please provide valid --image for classify")
                        continue

                    classify.run_classify(
                        container.embedding, container.cache, img_path, use_trained
                    )

                # ---------- CACHE ----------
                elif cmd["command"] == "cache":
                    print('cmd: ', cmd)
                    cache.run_cache(
                        container.cache,
                        sub_command=cmd["cache_action"],
                        key=cmd["cache_key"],
                    )

                else:
                    print(Msg.alert(f"Unknown command: {cmd['command']}"))

            except KeyboardInterrupt:
                print(Msg.info("\nExiting serve..."))
                break


if __name__ == "__main__":
    main()
