#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <limits.h>

#define N 10000  // The number of elements of the array is N * N

/**
 * Initializes the array and defines its initial uniform random initial state. Our array contains two states either 1 or -1 (atomic "spins").
 * 
 * In order to avoid checking for conditions in the parallel section of the program regarding the array limits the array is expanded by 2 on both 
 * dimensions.
 * 
 *      e.g.
 * 
 *          This is the initial 3x3 array:      The array created has is (3 + 2)x(3 + 2) and is the following:
 * 
 *                  0   1   2                                        0   1   2   3   4
 *                ┌───┬───┬───┐                                    ┌───┬───┬───┬───┬───┐
 *             0  │ 0 │ 1 │ 2 │                                 0  │ x │ 6 │ 7 │ 8 │ x │
 *                ├───┼───┼───┤                                    ├───┼───┼───┼───┼───┤
 *             1  │ 3 │ 4 │ 5 │                                 1  │ 2 │ 0 │ 1 │ 2 │ 0 │
 *                ├───┼───┼───┤                                    ├───┼───┼───┼───┼───┤
 *             2  │ 6 │ 7 │ 8 │                                 2  │ 5 │ 3 │ 4 │ 5 │ 3 │
 *                └───┴───┴───┘                                    ├───┼───┼───┼───┼───┤
 *                                                              3  │ 8 │ 6 │ 7 │ 8 │ 7 │
 *                                                                 ├───┼───┼───┼───┼───┤
 *                                                              4  │ x │ 0 │ 1 │ 2 │ x │
 *                                                                 └───┴───┴───┴───┴───┘
 * 
 *          In the matrix created 0 has it's neighbors next to it with 2 and 6 wrapping around. The x values are not used.
 * 
 * 
 * @param arr The pointer to the array that is being initialized
 */
void initializeArray(short int **arr){
    srand(time(NULL));

    // i and j start from 1 because the initial array is surrounded by the wrapping rowns and columns
    for (int i = 1; i <= N; i++){
        for (int j = 1; j <= N; j++){

            int rnd = rand() % 100;  // Get a double random number in (0,1) range
            
            // 0.5 is chosen so that +1 and -1 are 50% each
            if (rnd >= 50){

                // positive spin
                arr[i][j] = 1;  
                                
            } else {

                // negatine spin
                arr[i][j] = -1;
            }

            // Wrap around for rows: periodic boundary conditions
            if (i == 1){

                // If i == 1 it means that this is the 0 row of the initial array and must be wrapped
                // to the row N - 1 of the final array (see the example above)
                arr[N + 1][j] = arr[1][j];

            } else if (i == N){

                // If i == N it means that this is the N - 1 row of the initial array and must be wrapped
                // to the row 0 of the final array (see the example above)
                arr[0][j] = arr[N][j];
            }
                
            // Wrap around for cols: periodic boundary conditions
            if (j == 1){

                // If j == 1 it means that this is the 0 col of the initial array and must be wrapped
                // to the col N - 1 of the final array (see the example above)
                arr[i][N + 1] = arr[i][1];

            } else if (j == N){

                // If j == N it means that this is the N - 1 col of the initial array and must be wrapped
                // to the col 0 of the final array (see the example above)
                arr[i][0] = arr[i][N];
            }
        }
    }
}

/**
 * @brief Prints a 2D array 
 * 
 * @param arr The array to print
 */
void printArray(short int **arr){

    for (int i = 0; i < N + 2; i++){
        for (int j = 0; j < N + 2; j++){
            
            if (arr[i][j] < 0){
                printf("%hd ", arr[i][j]);
            
            } else {
                printf(" %hd ", arr[i][j]);
            }
        }

        printf("\n");
    }

    printf("\n\n\n");
}

/**
 * @brief Find the sign value of the input
 * 
 * @param sum The summation result to find the sign
 * @return 1 if sum is > 0, -1 else 
 */
short int sign(short int sum){
    return sum > 0 ? 1 : -1;
}

/**
 * @brief Sums all the lements of the array. This is used to detect convergence. If the value of the sum is the same
 * for multiple iterations there is a good possibility that the simulation has reached a repetitive state
 * 
 * @param arr The array to produce the sum
 * @return The sum of all the lements of the array 
 */
int summation(short int **arr){
    
    int sum = 0;
    
    for (int i = 1; i <= N; i++){
        for (int j = 1; j <= N; j++){
            sum += arr[i][j];
        }
    }

    return sum;
}

/**
 * @brief Run the simulation for the number of iterations specified. For every iteration all the elements (moments) of
 * the read array are read along with their neighbours, the sum of all the values is passed through the sigh function
 * and written to the write matrix. 
 * 
 * The read and write matrices are swapped and the next itteration begins 
 * 
 * @param read The matrix to read from
 * @param write The matrix to write to
 * @param iterations The number of iterations to run
 */
void simulateIsing(short int **read, short int **write, int iterations){

    // Holds the last 3 sums of the write matrix
    int previous_sum[3] = {INT_MAX, INT_MAX - 1, INT_MAX - 2};

    // The initial value is the summation of the read matrix
    previous_sum[0] = summation(read);

    // printf("Initial summation %d\n", previous_sum[0]);
    // printArray(read);

    // For every iteration ...
    for (int iteration = 0; iteration < iterations; iteration++){
        // For every row...
        for (int i = 1; i <= N; i++){
            // ... and every column
            for (int j = 1; j <= N; j++){
                
                // Calculate the new spin for each point
                int sum = read[i-1][j] + read[i][j-1] + read[i][j] + read[i+1][j] + read[i][j+1];

                write[i][j] = sign(sum);
            }
        }


        // Update the wrapping rows...
        for (int j = 1; j <= N; j++){
            write[N + 1][j] = write[1][j];
            write[0][j] = write[N][j];
        }

        // ... and columns as well
        for (int i = 1; i <= N; i++){
            write[i][N + 1] = write[i][1];
            write[i][0] = write[i][N];
        }

        // Count the new sum of the array
        previous_sum[iteration % 3] = summation(write);

        // printf("Iteration: %d, summation %d\n", iteration, previous_sum[iteration % 3]);
        // printArray(write);

        // If 3 sums in a row have the same value or if the same value appears every other iteration ...
        if (previous_sum[0] == previous_sum[2]) {

            // ... a stable state is reached so the iterations stop
            break;
        }
        
        // Swap the references for the two arrays so that on the next iteration we read from the previous write
        short int **tmp = read;
        read = write;
        write = tmp;
    }
}

/**
 * Calculates the elapse time
 *
 * @param begin the starting timestamp
 * @param end the ending timestamp
 * @return elapsed time in seconds
 */
double measureTime(struct timeval begin, struct timeval end) {
    long seconds;
    long microseconds;
    double elapsed;

    seconds = end.tv_sec - begin.tv_sec;
    microseconds = end.tv_usec - begin.tv_usec;
    elapsed = seconds + microseconds * 1e-6;

    return elapsed;
}

/**
 * @brief Main function runs the simulation until a stable state is reached or the number of iterations 
 * requested finish running.
 * 
 * To keep runnig evn if a stable state is reached change the break condition inside the simulateIsing
 * function to False
 * 
 */
int main(int argc, char **argv){

    // Vars needed for execution time measurement
    struct timeval begin, end;

    // The array is N + 2 size for the wrapping around on both dimensions.
    short int **array1 = (short int **) calloc ((N + 2), sizeof(short int *));

    for (int i = 0; i < N + 2; i++){
        array1[i] = (short int *) calloc ((N + 2), sizeof(short int));
    }

    // The array is N + 2 size for the wrapping around on both dimensions.
    short int **array2 = (short int **) calloc ((N + 2), sizeof(short int *));

    for (int i = 0; i < N + 2; i++){
        array2[i] = (short int *) calloc ((N + 2), sizeof(short int));
    }

    
    initializeArray(array1);
    // printArray(array1);


    printf("Starting Simulation\n");
    
    gettimeofday(&begin, 0);

    // Run the simulaation for 10 iterations
    simulateIsing(array1, array2, 50);

    gettimeofday(&end, 0);

    printf("Time for simulation: %.5f seconds.\n", measureTime(begin, end));

    // free memory
    for (int i = 0; i < N + 2; i++){
        free(array1[i]);
        free(array2[i]);
    }

    free(array1);
    free(array2);
    
    return 0;
}