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

bool grow_snake(t_snake* snake, int direction, int board_width, int board_height)
{
    int x = snake->head->x;
    int y = snake->head->y;

    switch (direction)
    {
        case UP:
            y++;
            break;
        case DOWN:
            y--;
            break;
        case LEFT:
            x--;
            break;
        case RIGHT:
            x++;
            break;
    }

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