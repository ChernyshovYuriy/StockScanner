from pathlib import Path

from colorama import Fore, Style


# ─────────────────────────────────────────────────────────────────────────────
# FUNDS FILE
# ─────────────────────────────────────────────────────────────────────────────


def read_funds(path: Path) -> float:
    """
    Read available capital from a plain-text file.
    The first non-blank, non-comment line is parsed as a float.
    Returns 0.0 if the file is missing, empty, or unparseable.
    """
    if not path.exists():
        print(f"{Fore.YELLOW}Funds file not found: {path}{Style.RESET_ALL}")
        return 0.0

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            value = float(line.replace(",", "").replace("$", ""))
            return value
        except ValueError:
            print(
                f"{Fore.YELLOW}Could not parse funds value '{line}' "
                f"in {path}{Style.RESET_ALL}"
            )
            return 0.0

    print(f"{Fore.YELLOW}Funds file is empty: {path}{Style.RESET_ALL}")
    return 0.0


def write_funds(path: Path, amount: float) -> None:
    """
    Overwrite the funds file with the updated remaining balance.
    Preserves any comment lines (starting with #) that were in the original.
    """
    comments: list[str] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("#"):
                comments.append(line)

    lines = comments + [f"{amount:.2f}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
