#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define N 2

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

            double rnd = (double) rand() / RAND_MAX;  // Get a double random number in (0,1) range
            
            // 0.5 is chosen so that +1 and -1 are 50% each
            if (rnd > 0.5){

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



__device__ short int sign(short int sum){
    return sum > 0 ? 1 : -1;
}

__global__ void simulateIsing(short int *arr1, short int *arr2, int iterations, int size){
    
    // Used for switching roles between the two arrays
    short int *read = arr1;
    short int *write = arr2;

    if (threadIdx.x == 0) {
        for (int i = 0; i < (size + 2) * (size + 2) ; i++) {
            printf("%hd ", read[i]);
        }
        printf("\n\n");
    }
    
    
    // Index for the flatted out 2D array
    int index = size + 3 + threadIdx.x + 2 * threadIdx.x / size;

    // Calculate the new spin for each point
    int sum = read[index - 1] + read[index + 1] + read[index - (size + 2)] + read[index + (size + 2)] + read[index];

    write[index] = sign(sum);

    // Synchronise the threads
    __syncthreads();

    read = arr2;
    write = arr1;
}

int main(int argc, char **argv){

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

    short int *d_array1;
    short int *d_array2;

    cudaMalloc((void**)&d_array1, sizeof(short int) * (N + 2) * (N + 2));
    cudaMalloc((void**)&d_array2, sizeof(short int) * (N + 2) * (N + 2));

    initializeArray(array1);
    printArray(array1);

    for (int i = 0; i < N + 2; i++) {
        cudaMemcpy(d_array1 + i * (N + 2), array1[i], sizeof(short int) * (N + 2), cudaMemcpyHostToDevice);
        cudaMemcpy(d_array2 + i * (N + 2), array2[i], sizeof(short int) * (N + 2), cudaMemcpyHostToDevice);
    }
    
    simulateIsing <<<1, N * N>>> (d_array1, d_array2, 10, N);

    for (int i = 0; i < N + 2; i++) {
        cudaMemcpy(array1[i], d_array1 + i * (N + 2), sizeof(short int) * (N + 2), cudaMemcpyDeviceToHost);
        cudaMemcpy(array2[i], d_array2 + i * (N + 2), sizeof(short int) * (N + 2), cudaMemcpyDeviceToHost);
    }

    printArray(array1);

    // free memory
    for (int i = 0; i < N + 2; i++){
        free(array1[i]);
        free(array2[i]);
    }

    free(array1);
    free(array2);
    
    return 0;
}