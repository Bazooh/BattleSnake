#include "game.h"

// t_board* init_board(int width, int height, int nb_snakes)
// {
//     t_board* board = malloc(sizeof(t_board));

//     board->width = width;
//     board->height = height;
//     board->nb_snakes = nb_snakes;
//     board->snakes_matrix = malloc(sizeof(int*) * width);
//     board->apples_matrix = malloc(sizeof(int*) * width);
//     board->snakes = malloc(sizeof(t_snake*) * 2);
//     board->winner = NO_WINNER;
//     board->finished = false;

//     for (int x = 0; x < width; x++)
//     {
//         board->snakes_matrix[x] = malloc(sizeof(int) * height);
//         board->apples_matrix[x] = malloc(sizeof(int) * height);
//         for (int y = 0; y < height; y++)
//         {
//             board->snakes_matrix[x][y] = 0;
//             board->apples_matrix[x][y] = 0;
//         }
//     }
//     for (int i = 0; i < nb_snakes; i++)
//     {
//         board->snakes[i] = malloc(sizeof(t_snake));
//         board->snakes[i]->head = malloc(sizeof(t_snake_part));
//         board->snakes[i]->tail = malloc(sizeof(t_snake_part));
//         board->snakes[i]->head->prev = NULL;
//         board->snakes[i]->head->next = board->snakes[i]->tail;
//         board->snakes[i]->tail->prev = board->snakes[i]->head;
//         board->snakes[i]->tail->next = NULL;
//         board->snakes[i]->health = 100;
//     }
//     board->snakes[0]->head->x = 2;
//     board->snakes[0]->head->y = 2;
//     board->snakes[0]->tail->x = 2;
//     board->snakes[0]->tail->y = 3;
//     board->snakes[1]->head->x = 7;
//     board->snakes[1]->head->y = 7;
//     board->snakes[1]->tail->x = 7;
//     board->snakes[1]->tail->y = 6;

//     return board;
// }

int main() { return 0; }

void free_board(t_board* board)
{
    for (int x = 0; x < board->width; x++)
    {
        free(board->apples_matrix[x]);
        free(board->snakes_matrix[x]);
    }
    free(board->apples_matrix);
    free(board->snakes_matrix);

    for (int i = 0; i < board->nb_snakes; i++)
        free_snake(board->snakes[i]);

    free(board->snakes);
    free(board);
}

t_board* copy_board(t_board* board)
{
    t_board* new_board = malloc(sizeof(t_board));

    new_board->width = board->width;
    new_board->height = board->height;
    new_board->nb_snakes = board->nb_snakes;
    new_board->winner = board->winner;
    new_board->finished = board->finished;

    new_board->snakes_matrix = malloc(sizeof(int*) * board->width);
    new_board->apples_matrix = malloc(sizeof(int*) * board->width);
    for (int x = 0; x < board->width; x++)
    {
        new_board->snakes_matrix[x] = malloc(sizeof(int) * board->height);
        new_board->apples_matrix[x] = malloc(sizeof(int) * board->height);
        for (int y = 0; y < board->height; y++)
        {
            new_board->snakes_matrix[x][y] = board->snakes_matrix[x][y];
            new_board->apples_matrix[x][y] = board->apples_matrix[x][y];
        }
    }

    new_board->snakes = malloc(sizeof(t_snake*) * board->nb_snakes);
    for (int i = 0; i < board->nb_snakes; i++)
        new_board->snakes[i] = copy_snake(board->snakes[i]);

    return new_board;
}

bool move_snakes(t_board* board, int *directions)
{
    for (int i = 0; i < board->nb_snakes; i++)
    {
        if (!grow_snake(board->snakes[i], directions[i], board->width, board->height))
        {
            board->finished = true;
            board->winner = board->winner == NO_WINNER ? !i : NO_WINNER;
        }
        
        if (board->snakes[i]->tail->x != board->snakes[i]->tail->prev->x || board->snakes[i]->tail->y != board->snakes[i]->tail->prev->y)
            board->snakes_matrix[board->snakes[i]->tail->x][board->snakes[i]->tail->y] = 0;
        suppress_tail(board->snakes[i]);
    }

    if (board->finished) return false;

    for (int i = 0; i < board->nb_snakes; i++)
    {
        if (board->snakes_matrix[board->snakes[i]->head->x][board->snakes[i]->head->y])
        {
            board->finished = true;
            board->winner = board->winner == NO_WINNER ? !i : NO_WINNER;
        }
    }

    if (board->finished) return false;

    return true;
}

bool heads_collide(t_board* board)
{
    for (int i = 0; i < board->nb_snakes; i++)
        for (int j = i + 1; j < board->nb_snakes; j++)
            if (board->snakes[i]->head->x == board->snakes[j]->head->x && board->snakes[i]->head->y == board->snakes[j]->head->y)
            {
                board->finished = true;

                int length_i = snake_length(board->snakes[i]);
                int length_j = snake_length(board->snakes[j]);

                if (length_i < length_j)
                    board->winner = j;
                else if (length_i > length_j)
                    board->winner = i;
                else
                    board->winner = NO_WINNER;
                
                return true;
            }

    return false;
}

t_board* next_board(t_board* board, int *directions)
{
    board = copy_board(board);

    for (int i = 0; i < board->nb_snakes; i++)
        (board->snakes[i]->health)--;
    
    if (!move_snakes(board, directions)) return board;

    if (heads_collide(board)) return board;

    for (int i = 0; i < board->nb_snakes; i++)
    {
        board->snakes_matrix[board->snakes[i]->head->x][board->snakes[i]->head->y] = 1;

        if (board->apples_matrix[board->snakes[i]->head->x][board->snakes[i]->head->y])
        {
            add_tail(board->snakes[i]);
            board->snakes[i]->health = 100;
            board->apples_matrix[board->snakes[i]->head->x][board->snakes[i]->head->y] = 0;
        }

        if (board->snakes[i]->health <= 0)
        {
            board->finished = true;
            board->winner = board->winner == NO_WINNER ? !i : NO_WINNER;
        }
    }

    return board;
}