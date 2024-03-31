import subprocess
import json
import socket
from tqdm import tqdm
from Constants import *
import matplotlib.pyplot as plt
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

current_loss = 0.
in_browser_first_time = True

def moving_average(arr, n):
    result = []
    window_sum = 0
    
    for i in range(n):
        window_sum += arr[i]
    
    result.append(window_sum / n)
    
    for i in range(n, len(arr)):
        window_sum += arr[i] - arr[i - n]
        result.append(window_sum / n)
    
    return result

def run_command(command, in_browser: bool = True):
    global in_browser_first_time
    
    if in_browser:
        go_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        go_socket.bind(('localhost', 8080))
        go_socket.listen(1)
        command += " --browser"
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('localhost', 8888))
    sock.listen(1)
    
    subprocess.Popen(command, shell=True)

    if in_browser:
        connection, _ = go_socket.accept()
        url = connection.recv(1024).decode('utf-8')
        connection.close()
        go_socket.close()
        
        game_id = url.split('game=')[1].split('&')[0]
        
        if in_browser_first_time:
            driver.get(url)
            with open("./evilbar.js", 'r') as js_file:
                driver.execute_script(js_file.read(), game_id)
            
            in_browser_first_time = False

        driver.execute_script(f"window.game_href = \"{url}\";")

    connection, _ = sock.accept()
    
    is_game_finished = False
    game_scores = [0.5] # * NOT SURE IF THIS IS CORRECT
    
    while not is_game_finished:
        raw_data = connection.recv(1024)
        
        raw_data = raw_data.decode('utf-8').split('\n')
        for line in raw_data:
            if len(line) <= 0:
                continue
            
            data = line.split('|')
            turn, is_game_finished, score = int(data[0]), data[1] == "True", float(data[2])
            game_scores.append(score)
        
        if in_browser:
            driver.execute_script(f"window.scores[\"{game_id}\"] = JSON.parse('{json.dumps(game_scores)}');")

            if is_game_finished:
                driver.execute_script(f"window.end_turn[\"{game_id}\"] = {turn - 1};")
    
    global current_loss
    current_loss = float(data[3])
    turns.append(turn)
    
    connection.close()
    sock.close()


command = "cd /Users/aymeric/rules && ./battlesnake play -W 11 -H 11 --url http://0.0.0.0:8000 --name UCT_train --url http://0.0.0.0:8001 --name UCT"

turns = []

service = Service(executable_path="/Users/aymeric/Downloads/chromedriver-mac-arm64/chromedriver")
options = Options()
options.add_experimental_option("detach", True)
driver = webdriver.Chrome(service=service, options=options)

pbar = tqdm(range(N_RUN))
for i in pbar:
    desc = f'turn {turns[-1]:2}, loss {current_loss:.4f}' if len(turns) > 0 else 'turn --, loss -.----'
    pbar.set_description(desc)
    
    in_browser = N_GAMES_IN_BROWSER > 0 and i % (N_RUN // N_GAMES_IN_BROWSER) == 0
    run_command(command, in_browser)

# driver.quit()

if N_RUN > MOVING_AVERAGE_WINDOW:
    plt.plot(moving_average(turns, MOVING_AVERAGE_WINDOW))
    plt.show()