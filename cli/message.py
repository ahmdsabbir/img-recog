class Message:
    _YELLOW = "\033[33m"
    _CYAN = "\033[36m"
    _RED = "\033[31m"
    _RESET = "\033[0m"


    def info(self, msg: str):
        return f"{self._CYAN}{msg}{self._RESET}"

    def alert(self, msg: str):
        return f"{self._RED}{msg}{self._RESET}"

    def highlight(self, msg: str):
        return f"{self._YELLOW}{msg}{self._RESET}"

    def neutral(self, msg: str):
        return msg