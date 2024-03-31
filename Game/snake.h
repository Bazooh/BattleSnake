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
} t_snake;

void add_tail(t_snake* snake);
void suppress_tail(t_snake* snake);
bool grow_snake(t_snake* snake, int direction, int board_width, int board_height);
void free_snake(t_snake* snake);
t_snake* copy_snake(t_snake* snake);
int snake_length(t_snake* snake);

#endif // SNAKE_H