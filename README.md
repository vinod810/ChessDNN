# DNN based Chess Engine

The objectives of this project are to demonstrate 1) chess position evaluation using a deep neural network (DNN) and 2) a chess engine that can play at a strength equal to an experienced club player, i.e., about 1500 ELO rating strengt..

## Description

The project used about 100 million chess positions to train a DNN that can evaluate a chess position. The chess engine uses the chess position evaluation provided by the DNN to decide the best move by evaluating and ranking the resulting chess positions for each legal move. 

The Python-based chess engine that combines classical alpha–beta negamax search with modern heuristics. The engine speaks the UCI (Universal Chess Interface) protocol and can be used with chess GUIs such as PyChess, as well as other chess GUIs that support UCI. The key features of the chess engine are:
* UCI-compatible chess engine
* Negamax + Alpha–Beta pruning
* Iterative Deepening
* Quiescence Search
* Killer Move heuristic
* History heuristic
* Late Move Reductions (LMR)
* Null Move Pruning
* Aspiration Windows
* Time-controlled search
* Transposition Tables

## Getting Started
* Clone the repository to your computer
* Set up the conda environment using the environment.yml file
  * Build DNN model (optional). Note: You may skip this step and use the prebuilt model/model.keras
    * Download the most recent chess games files from https://database.lichess.org/ in pgn.zst format
    * Convert the pgn.zst files into multiple (about 100) one-hot encoded files by running the command 'python prepare_data.py filename.pgn.zst'.  Note: the conversion will take 2 to 3 days to complete.
    * Generate the DNN model by running the command 'python build_model.py'
  * Test the DNN position evaluation (optional). Run the command 'python predict_score.py'
  * Test the best move generation (optional). Run the command 'python best_move.py'

## Executing program
* You can play against this chess program using PyChess GUI.
  * Install the GUI using the command 'sudo apt install pychess'
  * Start PyChess by running the command 'pychess'
  * Configure PyChess to use this program as an engine.
    * Edit->Engines>New. Then browse to the cloned directory and select uci_engine.py
  * You are now ready to play against the engine. Good luck!

## Authors
Eapen Kuruvilla 
https://www.linkedin.com/in/eapenkuruvilla/

## License
This project is free to use, modify, and redistribute for any purpose, including commercial use. It is provided **“as is”**, without any warranty of any kind, and the authors assume no responsibility or liability 
for any damages arising from its use.









