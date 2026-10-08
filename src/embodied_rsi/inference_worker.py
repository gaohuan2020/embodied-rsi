"""JSON-lines isolated RSI model process; logs go to stderr."""
import contextlib
import json
import sys

from .storage import dumps


def main():
    with contextlib.redirect_stdout(sys.stderr):
        from .policies import InProcessRSI
        policy = InProcessRSI(sys.argv[1], revision=sys.argv[2] if len(sys.argv) > 2 else None)
    print(dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                p = policy.predict(request["state"], request["criteria"])
            print(dumps({"probabilities": p.tolist()}), flush=True)
        except Exception as exc:  # noqa: BLE001 -- process boundary returns structured errors
            print(dumps({"error": str(exc)[:1000]}), flush=True)


if __name__ == "__main__":
    main()
