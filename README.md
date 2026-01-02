# 🧠♟️ NeuroFish
NeuroFish is a hybrid chess engine that combines classical alpha–beta search techniques with selective neural network evaluation to produce strong, explainable move choices within fixed time constraints.

Unlike pure neural engines or purely handcrafted evaluators, NeuroFish blends the best of both worlds:
a fast, well-optimized negamax search core enhanced by a DNN-based positional evaluator that is invoked only when it is most informative.

The objectives of this project are to demonstrate 1) chess position evaluation using a deep neural network (DNN) and 2) a chess engine that can play at aen expert level strength, i.e., above 2000 ELO rating strength.

## Description
The project used about 40 million chess positions to train a DNN that can evaluate a chess position. The chess engine uses the chess position evaluation provided by the DNN to decide the best move by evaluating and ranking the resulting chess positions for each legal move. 

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

## Executing program
* You can play against this chess program using PyChess or Scid Chess GUI.
  * Install the GUI using the command 'sudo apt install pychess'
  * Start PyChess by running the command 'pychess'
  * Configure PyChess to use this program as an engine.
    * Edit->Engines>New. Then browse to the cloned directory and select uci_engine.py
  * You are now ready to play against the engine. Good luck!

## Authors
Eapen Kuruvilla 
https://www.linkedin.com/in/eapenkuruvilla/

## License
This project is free to use, modify, and redistribute for any purpose, including commercial use. It is provided as is, without any warranty of any kind, and the authors assume no responsibility or liability 
for any damages arising from its use.










