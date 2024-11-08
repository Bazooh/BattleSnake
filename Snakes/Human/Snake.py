import curses
from Snakes.Snake import Snake


# DOES NOT WORK

class HumanSnake(Snake):
    color: str = "#f57f18"
    head: str = "scarf"
    tail: str = "rbc-necktie"
    
    def __init__(self, **kwargs):
        self.color = kwargs.get("color", self.color)


    def start(self, game_state: dict):
        pass
    

    def end(self, game_state: dict):
        pass
    

    def wait_for_key_press(self, screen: curses._CursesWindow) -> str:
        key = screen.getch()

        if key == curses.KEY_RIGHT:
            return "right"
        elif key == curses.KEY_LEFT:
            return "left"
        elif key == curses.KEY_UP:
            return "top"
        elif key == curses.KEY_DOWN:
            return "down"
        
        assert False, f"Unknown key: {key}"


    def move(self, game_state: dict) -> dict:
        move = curses.wrapper(self.wait_for_key_press)
        
        print(f"Move: {move}")
        return {"move": move}
