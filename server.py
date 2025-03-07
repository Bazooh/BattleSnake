import logging
import os
import typing

from flask import Flask
from flask import request


def run_server(handlers: typing.Dict, port=8000):
    app = Flask("Battlesnake")

    @app.get("/")
    def on_info():
        return handlers["info"]()

    @app.post("/start")
    def on_start():
        game_state = request.get_json()
        handlers["start"](game_state)
        return "ok"

    @app.post("/move")
    def on_move():
        game_state = request.get_json()
        response: dict = handlers["move"](game_state)

        # if 'score' in response:
        #     # Create a new Chrome browser instance
        #     driver = webdriver.Chrome()

        #     # Get the currently active tab/window handle
        #     current_handle = driver.current_window_handle

        #     # Switch the focus to the current tab
        #     driver.switch_to.window(current_handle)

        #     # Execute JavaScript code to add content to the current page
        #     script = """
        #     var newDiv = document.createElement('div');
        #     newDiv.textContent = 'This is a dynamically added div.';
        #     document.body.appendChild(newDiv);
        #     """
        #     driver.execute_script(script)

        #     # Close the browser
        #     driver.quit()

        return response

    @app.post("/end")
    def on_end():
        game_state = request.get_json()
        handlers["end"](game_state)
        return "ok"

    # @app.after_request
    # def identify_server(response):
    #     response.headers.set(
    #         "server", "battlesnake/github/starter-snake-python"
    #     )
    #     return response

    host = "0.0.0.0"
    port = int(os.environ.get("PORT", str(port)))

    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    print(f"\nRunning Battlesnake at http://{host}:{port}")
    app.run(host=host, port=port)
