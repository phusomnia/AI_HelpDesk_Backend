import logging
import sys

class CustomFormatter(logging.Formatter):
    # Bảng mã màu ANSI
    GREY = "\033[0;37m"    # Màu xám cho message thông thường
    BLUE = "\033[0;34m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    RED = "\033[1;31m"     # Đỏ thường cho ERROR
    BOLD_RED = "\033[1;31;1m" # Đỏ đậm cho CRITICAL
    
    CYAN = "\033[0;36m"    # Màu cho [Name]
    PURPLE = "\033[0;35m"  # Màu cho [Time]
    RESET = "\033[0m"

    def format(self, record):
        # 1. Định nghĩa màu cho LevelName
        level_colors = {
            logging.DEBUG: self.BLUE,
            logging.INFO: self.GREEN,
            logging.WARNING: self.YELLOW,
            logging.ERROR: self.RED,
            logging.CRITICAL: self.BOLD_RED
        }
        lvl_col = level_colors.get(record.levelno, self.RESET)

        # 2. Logic đổi màu cho Message:
        # Nếu là ERROR (40) hoặc CRITICAL (50), dùng màu của level đó (màu đỏ).
        # Nếu thấp hơn (INFO, DEBUG, WARNING), dùng màu xám (GREY).
        if record.levelno >= logging.ERROR:
            msg_col = lvl_col
        else:
            msg_col = self.GREY

        # 3. Định nghĩa Format chuỗi log
        log_fmt = (
            f"{lvl_col}%(levelname)-1s{self.RESET}: "       # Level (Màu theo cấp độ)
            f"[{self.GREY}%(name)-1s{self.RESET}] - "       # Name (Màu Cyan)
            f"{msg_col}%(message)s{self.RESET} - "          # Message (Xám hoặc Đỏ)
            f"[{self.PURPLE}%(asctime)s{self.RESET}]"       # Time (Màu Purple)
        )

        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

class Logger:
    _instances = {}

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        if name not in cls._instances:
            logger = logging.getLogger(name)
            logger.propagate = False
            logger.setLevel(logging.DEBUG)
            
            if logger.handlers:
                logger.handlers = []
                
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(CustomFormatter())
            logger.addHandler(handler)
            
            cls._instances[name] = logger
            
        return cls._instances[name]

def get_logger(name: str = "App"):
    return Logger.get_logger(name)
