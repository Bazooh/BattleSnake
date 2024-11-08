window.games_infos = {}
window.game_href = ""

function program(_window, game_id) {

    let doneLoadingGame = false;
    let doneLoadingNextGame = false;

    function updateEvaluationBar(turn=0) {
        let game_scores = game_id in window.games_infos ? window.games_infos[game_id]["scores"] : [0];
        let game_trees_size = game_id in window.games_infos ? window.games_infos[game_id]["trees_size"] : 0;

        let new_score = turn < game_scores.length ? game_scores[turn] : game_scores[game_scores.length - 1];
        let new_tree_size = turn < game_trees_size.length ? game_trees_size[turn] : game_trees_size[game_trees_size.length - 1];

        const bar = _window.document.getElementById('evilbar-fill');
        const value = _window.document.getElementById('evilbar-value');
        bar.style.height = `${(new_score + 1) * 50}%`;
        value.innerHTML = `${new_score.toFixed(2)}\n${new_tree_size}`;
    }

    function createEvilBar() {
        const style = _window.document.createElement('style');
        style.textContent = `
            .bar {
                width: 40px;
                height: 95%;
                background-color: #010101;
                border-radius: 10px;
                overflow: hidden;
                transform: rotate(180deg);
                margin-left: 20px;
            }

            .bar-fill {
                width: 100%;
                background-color: #f57f18;
                transition: height 0.5s;
            }

            .bar-value {
                position: absolute;
                bottom: 0;
                width: 100%;
                text-align: center;
                color: #f1f1f1;
                font-weight: bold;
                padding-top: 10px;
                transform: rotate(180deg);
            }
        `;
        _window.document.head.append(style);

        const bar = _window.document.createElement('div');
        bar.className = 'bar';
        bar.innerHTML = `
            <div class="bar-fill" id="evilbar-fill"></div>
            <div class="bar-value" id="evilbar-value"></div>
        `;
        return bar;
    }

    function onLoad(elem) {
        elem.prepend(createEvilBar());
        updateEvaluationBar();
        turnObserver.observe(_window.document.querySelector('.basis-1\\/2.pl-2'), { characterData: true, subtree: true});
    }

    const loadingObserver = new MutationObserver((mutations, observer) => {

        _window.console.log("Loading Observer Callback triggered");

        mutations.forEach((mutation) => {
            if (mutation.type === 'childList') {
                const main_div = _window.document.querySelector('main > div > div > div');

                if (!main_div || doneLoadingGame) { return; }
                
                let num_child = 0;
                let child = main_div.firstChild;
                while (child) {
                    num_child++;
                    child = child.nextSibling;
                }

                if (num_child >= 2) {
                    doneLoadingGame = true;
                    observer.disconnect();

                    onLoad(main_div);
                }
            }
        });
    });

    const turnObserver = new MutationObserver((mutations, observer) => {
        mutations.forEach((mutation) => {
            if (mutation.type === 'characterData') {
                let turn = parseInt(mutation.target.textContent);

                updateEvaluationBar(turn + 1);
                if (doneLoadingNextGame) { return; }

                console.log(turn, window.games_infos[game_id]["end_turn"])

                if (turn == window.games_infos[game_id]["end_turn"]) {
                    console.log("load...")
                    setTimeout(function() {
                        let new_window = _window.open(window.game_href);
                        if (new_window)
                        {
                            setTimeout(function() {
                                program(new_window.window, window.game_href.match(/game=(.*?)&/)[1]);
                            }, 1000);

                            doneLoadingNextGame = true;
                        }
                    }, 1000);
                }
            }
        });
    });

    _window.console.log(_window.document.body);
    loadingObserver.observe(_window.document.body, { childList: true, subtree: true });
}

program(window, arguments[0])