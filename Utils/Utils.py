

def cprint(msg: str, color: str = "blue", **kwargs) -> str:
    if color == "blue": print("\033[34m" + msg + "\033[0m", **kwargs)
    elif color == "red": print("\033[31m" + msg + "\033[0m", **kwargs)
    elif color == "green": print("\033[32m" + msg + "\033[0m", **kwargs)
    elif color == "yellow": print("\033[33m" + msg + "\033[0m", **kwargs)
    elif color == "purple": print("\033[35m" + msg + "\033[0m", **kwargs)
    elif color == "cyan": print("\033[36m" + msg + "\033[0m", **kwargs)
    else: raise ValueError(f"Invalid info color: `{color}`")


