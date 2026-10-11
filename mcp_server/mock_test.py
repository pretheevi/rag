import json

from mcp_server.tools.search_testcases import search_testcases


def main() -> None:
    print("Semantic testcase search. Enter a query, or 'q' to quit.")

    while True:
        try:
            query = input("\nSearch query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return

        if query.casefold() in {"q", "quit", "exit"}:
            print("Exiting.")
            return
        if not query:
            continue

        try:
            results = search_testcases(query=query)
        except (RuntimeError, ValueError) as exc:
            print(f"Search failed: {exc}")
            continue

        if not results:
            print("No matching test cases found.")
            continue

        print(json.dumps(results, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
