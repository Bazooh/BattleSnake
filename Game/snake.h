#ifndef SNAKE_H
#define SNAKE_H

#include <stdlib.h>
#include <stdbool.h>

enum e_direction
{
    UP,
    RIGHT,
    DOWN,
    LEFT
};

enum e_local_action
{
    LOCAL_LEFT,
    STRAIGHT,
    LOCAL_RIGHT
};

#define N_LOCAL_ACTIONS 3
#define N_GLOBAL_ACTIONS 4

typedef struct s_snake_part
{
    int x;
    int y;
    struct s_snake_part *next;
    struct s_snake_part *prev;
} t_snake_part;

typedef struct s_snake
{
    t_snake_part *head;
    t_snake_part *tail;
    int health;
    int global_direction;
    bool playable_actions[N_LOCAL_ACTIONS];
} t_snake;

void add_tail(t_snake* snake);
void suppress_tail(t_snake* snake);
bool grow_snake(t_snake* snake, int direction, int board_width, int board_height);
void free_snake(t_snake* snake);
t_snake* copy_snake(t_snake* snake);
int snake_length(t_snake* snake);
bool compute_playable_actions(t_snake* snake, int** snakes_matrix, t_snake* *snakes, int nb_snakes, int board_width, int board_height);
int move_snake(int global_direction, int local_direction, int board_width, int board_height, int* x, int* y);

#endif // SNAKE_H