"""TEMP helper: strip trailing whitespace on lines outside string literals."""
import io
import sys
import tokenize


def full_tokens(src):
    return [
        (t.type, t.string)
        for t in tokenize.generate_tokens(io.StringIO(src).readline)
    ]


def fix(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        src = f.read()

    in_str = set()
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.type == tokenize.STRING and t.start[0] != t.end[0]:
            in_str.update(range(t.start[0], t.end[0] + 1))

    lines = src.splitlines(keepends=True)
    changed = False
    for i, line in enumerate(lines, 1):
        if i in in_str:
            continue
        stripped = line.rstrip(" \t")
        if stripped != line:
            lines[i - 1] = stripped
            changed = True

    if not changed:
        print(f"no-op   {path}")
        return

    new_src = "".join(lines)
    if full_tokens(new_src) != full_tokens(src):
        raise SystemExit(f"TOKEN MISMATCH in {path}: whitespace strip changed code!")
    compile(new_src, path, "exec")

    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(new_src)
    print(f"fixed   {path}")


def main():
    for path in sys.argv[1:]:
        fix(path)
    print("done")


if __name__ == "__main__":
    main()
