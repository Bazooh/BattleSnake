#ifndef GAME_H
#define GAME_H

#include "snake.c"

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

#define NO_WINNER 0.5f

typedef struct s_board
{
    int width;
    int height;
    int nb_snakes;
    int **snakes_matrix;
    int **apples_matrix;
    t_snake* *snakes;
    float winner;
    bool finished;
} t_board;

void free_board(t_board* board);
bool move_snakes(t_board* board, int *directions);
bool heads_collide(t_board* board);
t_board* next_board(t_board* board, int *directions);

#endif // GAME_H