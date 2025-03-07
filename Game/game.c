// To compile:
// gcc -shared -o Game/game.so -fPIC Game/game.c


#include "game.h"


#define clamp(x, min, max) ((x) <= (min) ? (min) : ((x) >= (max) ? (max) : (x)))


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

    if (board->finished) return;

    // for (int snake = 0; snake < board->nb_snakes; snake++)
    // {
    //     for (int i = 0; i < 2*VIEW_SIZE + 1; i++)
    //         free(board->convs[snake][i]);
    //     free(board->convs[snake]);
    // }
    // free(board->convs);

    // for (int snake = 0; snake < board->nb_snakes; snake++)
        // free(board->aids[snake]);
    // free(board->aids);
}


t_board* copy_board(t_board* board)
{
    t_board* new_board = malloc(sizeof(t_board));

    new_board->width = board->width;
    new_board->height = board->height;
    new_board->nb_snakes = board->nb_snakes;
    new_board->winner = board->winner;
    new_board->finished = board->finished;
    new_board->turn = board->turn;

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
    for (int snake = 0; snake < board->nb_snakes; snake++)
        new_board->snakes[snake] = copy_snake(board->snakes[snake]);

    return new_board;
}


bool move_snakes(t_board* board, int *local_directions)
{
    for (int snake = 0; snake < board->nb_snakes; snake++)
    {
        if (!grow_snake(board->snakes[snake], local_directions[snake], board->width, board->height))
        {
            board->finished = true;
            board->winner = board->winner == NO_WINNER ? (snake == 0 ? OTHER_PLAYER : MAIN_PLAYER) : NO_WINNER;
        }
        
        if (board->snakes[snake]->tail->x != board->snakes[snake]->tail->prev->x || board->snakes[snake]->tail->y != board->snakes[snake]->tail->prev->y)
            board->snakes_matrix[board->snakes[snake]->tail->x][board->snakes[snake]->tail->y] = 0;
        suppress_tail(board->snakes[snake]);
    }

    if (board->finished) return false;

    for (int snake = 0; snake < board->nb_snakes; snake++)
    {
        if (board->snakes_matrix[board->snakes[snake]->head->x][board->snakes[snake]->head->y])
        {
            board->finished = true;
            board->winner = board->winner == NO_WINNER ? (snake == 0 ? OTHER_PLAYER : MAIN_PLAYER) : NO_WINNER;
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
                    board->winner = -1;
                else if (length_i > length_j)
                    board->winner = 1;
                else
                    board->winner = NO_WINNER;
                
                return true;
            }

    return false;
}


float distance_to_nearest_apple(t_board* board, t_snake* snake, int max_distance, int local_direction)
{
    int x = snake->head->x;
    int y = snake->head->y;
    
    move_head(snake->global_direction, local_direction, board->width, board->height, &x, &y);

    int min_distance = max_distance;

    for (int i = 0; i < board->width; i++)
        for (int j = 0; j < board->height; j++)
            if (board->apples_matrix[i][j])
            {
                int distance = (x - i)*(x - i) + (y - j)*(y - j);
                if (distance < min_distance)
                    min_distance = distance;
            }

    return (float)sqrt((double)min_distance / max_distance);
}


void create_tensor(t_board* board)
{
    board->convs = malloc(board->nb_snakes * sizeof(float**));
    for (int snake = 0; snake < board->nb_snakes; snake++)
    {
        board->convs[snake] = malloc(sizeof(float*) * (2*VIEW_SIZE + 1));
        for (int i = 0; i < 2*VIEW_SIZE + 1; i++)
            board->convs[snake][i] = calloc(2*VIEW_SIZE + 1, sizeof(float));
    }

    board->aids = malloc(sizeof(float*) * board->nb_snakes);
    for (int snake = 0; snake < board->nb_snakes; snake++)
        board->aids[snake] = calloc(AID_SIZE, sizeof(float));

    for (int snake = 0; snake < board->nb_snakes; snake++)
    {
        int pos_x = board->snakes[snake]->head->x;
        int pos_y = board->snakes[snake]->head->y;

        if (!board->snakes[snake]->head->next) return;

        int direction_x = pos_x - board->snakes[snake]->head->next->x;
        int direction_y = pos_y - board->snakes[snake]->head->next->y;

        for (int i = -VIEW_SIZE; i <= VIEW_SIZE; i++)
        {
            for (int j = -VIEW_SIZE; j <= VIEW_SIZE; j++)
            {
                int dx = direction_x*j + direction_y*i;
                int dy = direction_y*j - direction_x*i;

                int x = pos_x + dx;
                int y = pos_y + dy;

                if (x < 0 || x >= board->width || y < 0 || y >= board->height)
                    board->convs[snake][i + VIEW_SIZE][j + VIEW_SIZE] = 1;
                else if (board->snakes_matrix[x][y])
                    board->convs[snake][i + VIEW_SIZE][j + VIEW_SIZE] = 1;
                else
                    board->convs[snake][i + VIEW_SIZE][j + VIEW_SIZE] = 0;
            }
        }
    }

    int max_distance = board->width*board->width + board->height*board->height;
    int lengths[2] = {snake_length(board->snakes[0]), snake_length(board->snakes[1])};
    for (int snake = 0; snake < board->nb_snakes; snake++)
    {
        board->aids[snake][0] = clamp(lengths[snake] - lengths[1 - snake], -5, 5) / 5.0f;
        for (int local_action = 0; local_action < 3; local_action++)
            board->aids[snake][1 + local_action] = board->snakes[snake]->playable_actions[local_action] ?
                distance_to_nearest_apple(board, board->snakes[snake], max_distance, local_action) : 0;
    }
}


void compute_snakes_playable_actions(t_board* board)
{
    for (int snake = 0; snake < board->nb_snakes; snake++)
        if (!compute_playable_actions(board->snakes[snake], board->snakes_matrix, board->snakes, board->nb_snakes, board->width, board->height))
        {
            board->finished = true;
            board->winner = board->winner == NO_WINNER ? (snake == 0 ? OTHER_PLAYER : MAIN_PLAYER) : NO_WINNER;
        }
}


t_board* next_board(t_board* board, int *local_directions)
{
    board = copy_board(board);
    board->turn++;

    for (int snake = 0; snake < board->nb_snakes; snake++)
        (board->snakes[snake]->health)--;
    
    if (!move_snakes(board, local_directions))
        return board;

    if (heads_collide(board))
        return board;

    for (int snake = 0; snake < board->nb_snakes; snake++)
    {
        board->snakes_matrix[board->snakes[snake]->head->x][board->snakes[snake]->head->y] = 1;

        if (board->apples_matrix[board->snakes[snake]->head->x][board->snakes[snake]->head->y])
        {
            add_tail(board->snakes[snake]);
            board->snakes[snake]->health = 100;
            board->apples_matrix[board->snakes[snake]->head->x][board->snakes[snake]->head->y] = 0;
        }

        if (board->snakes[snake]->health <= 0)
        {
            board->finished = true;
            board->winner = board->winner == NO_WINNER ? (snake == 0 ? OTHER_PLAYER : MAIN_PLAYER) : NO_WINNER;
        }
    }

    if (!board->finished)
        compute_snakes_playable_actions(board);
    
    if (!board->finished)
        create_tensor(board);

    return board;
}