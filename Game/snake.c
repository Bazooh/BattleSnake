#include "snake.h"


void add_head(t_snake* snake, int x, int y)
{
    t_snake_part* new_head = malloc(sizeof(t_snake_part));

    new_head->x = x;
    new_head->y = y;
    new_head->prev = NULL;
    new_head->next = snake->head;
    
    snake->head->prev = new_head;
    snake->head = new_head;
}


void add_tail(t_snake* snake)
{
    t_snake_part* new_part = malloc(sizeof(t_snake_part));

    new_part->x = snake->tail->x;
    new_part->y = snake->tail->y;
    new_part->prev = snake->tail;
    new_part->next = NULL;
    
    snake->tail->next = new_part;
    snake->tail = new_part;
}


void suppress_tail(t_snake* snake)
{
    t_snake_part* tail = snake->tail;

    snake->tail = tail->prev;
    free(tail);
    snake->tail->next = NULL;
}


int move_head(int global_direction, int local_direction, int board_width, int board_height, int* x, int* y)
{
    int new_global_direction = (global_direction + local_direction - 1) & 0b11;

    switch (new_global_direction)
    {
        case UP:
            (*y)++;
            break;
        case DOWN:
            (*y)--;
            break;
        case LEFT:
            (*x)--;
            break;
        case RIGHT:
            (*x)++;
            break;
    }

    return new_global_direction;
}


bool grow_snake(t_snake* snake, int local_direction, int board_width, int board_height)
{
    int x = snake->head->x;
    int y = snake->head->y;

    snake->global_direction = move_head(snake->global_direction, local_direction, board_width, board_height, &x, &y);

    if (x < 0 || x >= board_width || y < 0 || y >= board_height)
        return false;

    add_head(snake, x, y);

    return true;
}


void free_snake(t_snake* snake)
{
    t_snake_part* part = snake->head;

    while (part != NULL)
    {
        t_snake_part* next = part->next;
        free(part);
        part = next;
    }

    free(snake);
}


t_snake_part* copy_snake_part(t_snake_part* part)
{
    t_snake_part* new_part = malloc(sizeof(t_snake_part));

    new_part->x = part->x;
    new_part->y = part->y;
    new_part->next = NULL;
    new_part->prev = NULL;

    return new_part;
}


t_snake* copy_snake(t_snake* snake)
{
    t_snake* new_snake = malloc(sizeof(t_snake));

    new_snake->health = snake->health;
    new_snake->tail = copy_snake_part(snake->tail);
    new_snake->head = new_snake->tail;
    new_snake->global_direction = snake->global_direction;
    for (t_snake_part* part = snake->tail->prev; part != NULL; part = part->prev)
        add_head(new_snake, part->x, part->y);
    
    return new_snake;
}


int snake_length(t_snake* snake)
{
    int length = 0;

    for (t_snake_part* part = snake->tail; part != NULL; part = part->prev)
        length++;

    return length;
}


bool is_colliding_with_removing_tail(t_snake* *snakes, int nb_snakes, int x, int y)
{
    for (int snake = 0; snake < nb_snakes; snake++)
        if (x == snakes[snake]->tail->x && y == snakes[snake]->tail->y)
        {
            // There can be a stack of tail on the same cell, we need to check if the tail is the last one
            return (
                snakes[snake]->tail->prev->x != x ||
                snakes[snake]->tail->prev->y != y
            );
        }
    
    return false;
}


bool compute_playable_actions(t_snake* snake, int** snakes_matrix, t_snake* *snakes, int nb_snakes, int board_width, int board_height)
{
    bool playable = false;

    for (int local_direction = 0; local_direction < N_LOCAL_ACTIONS; local_direction++)
    {
        int x = snake->head->x;
        int y = snake->head->y;

        move_head(snake->global_direction, local_direction, board_width, board_height, &x, &y);

        snake->playable_actions[local_direction] = (x >= 0 && x < board_width && y >= 0 && y < board_height);
        
        if (snake->playable_actions[local_direction] && snakes_matrix[x][y])
            // At this point the snake is colliding with a snake
            // We need to check if it is colliding with a tail that will be remove in the next turn
            snake->playable_actions[local_direction] = is_colliding_with_removing_tail(snakes, nb_snakes, x, y);
        
        playable |= snake->playable_actions[local_direction];
    }

    return playable;
}