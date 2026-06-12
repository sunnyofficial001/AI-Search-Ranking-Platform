#!/usr/bin/env python3
"""Run a quick demo:
- generate synthetic demo data
- optionally POST a sample search request to the running dev server (http://localhost:3000)
"""

import json
import time

from generate_demo_data import generate_all

DEFAULT_SERVER = "http://localhost:3000"


def post_search(server=DEFAULT_SERVER, query="wireless headphones"):
    import urllib.request

    url = server.rstrip("/") + "/api/search"
    data = json.dumps({"query": query, "weights": {}}).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = resp.read().decode("utf-8")
            print("Server response:")
            print(body)
    except Exception as e:
        print(f"Failed to call {url}: {e}")


def main():
    print("Generating synthetic demo dataset...")
    generate_all()
    print("Waiting 0.5s for IO...")
    time.sleep(0.5)

    print("\nAttempting example search request to dev server (http://localhost:3000)...")
    post_search()


if __name__ == "__main__":
    main()
