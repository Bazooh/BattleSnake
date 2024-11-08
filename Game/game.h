#ifndef GAME_H
#define GAME_H

#include "snake.c"

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>

#define NO_WINNER 0
#define MAIN_PLAYER 1
#define OTHER_PLAYER -1

#define VIEW_SIZE 3
#define AID_SIZE 4

typedef struct s_board
{
    int width;
    int height;
    int nb_snakes;
    int **snakes_matrix;
    int **apples_matrix;
    t_snake* *snakes;
    int winner;
    bool finished;
    int turn;
    float*** convs;
    float** aids;
} t_board;

void free_board(t_board* board);
bool move_snakes(t_board* board, int *directions);
bool heads_collide(t_board* board);
t_board* next_board(t_board* board, int *directions);

#endif // GAME_H