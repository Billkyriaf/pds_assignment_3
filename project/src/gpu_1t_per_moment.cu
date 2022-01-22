#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define N 8

/**
 * @brief Initializes the array and defines its initial uniform random initial state. Our array contains two states either 1 or -1 (atomic "spins").
 * 
 * In order to avoid checking for conditions in the parallel section of the program regarding the array limits the array is expanded by 2 on both 
 * dimensions.
 * 
 *      e.g.
 * 
 *      
 * 
 * @param arr The array that should be initialized
 */
void initializeArray(short int **arr){
    srand(time(NULL));

    for(int i = 1; i <= N; i++){
        for(int j = 1; j <= N; j++){
            double rnd = (double) rand() / RAND_MAX;  // Get a double random number in (0,1) range

            if (rnd > 0.5){
                arr[i][j] = 1;  
                
                // Wrap around for rows: periodic boundary conditions
                if (i == 1){
                    arr[N + 1][j] = arr[1][j];

                } else if (i == N){
                    arr[0][j] = arr[N][j];
                }
                
                // Wrap around for cols: periodic boundary conditions
                if (j == 1){
                    arr[i][N + 1] = arr[i][1];

                } else if (j == N){
                    arr[i][0] = arr[i][N];
                }
                                
            } else {
                arr[i][j] = -1;

                // Wrap around for rows: periodic boundary conditions
                if (i == 1){
                    arr[N + 1][j] = arr[1][j];

                } else if (i == N){
                    arr[0][j] = arr[N][j];
                }
                
                // Wrap around for cols: periodic boundary conditions
                if (j == 1){
                    arr[i][N + 1] = arr[i][1];

                } else if (j == N){
                    arr[i][0] = arr[i][N];
                }
            }
        }
    }
}


/**
 * @brief Prints an array 
 * 
 * @param arr The array that should be printed
 */
void printArray(short int **arr){
    for(int i = 0; i < N + 2; i++){
        for(int j = 0; j < N + 2; j++){
            if(arr[i][j] < 0) {
                printf("%hd ", arr[i][j]);
            }
            else {
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
    
    // Index for the flatted out 2D array
    int index = size + 3 + threadIdx.x + 2 * threadIdx.x / size;

    // Calculate the new spin for each point
    int sum = read[index - 1] + read[index + 1] + read[index - (size + 2)] + read[index + (size + 2)] + read[index];

    write[i][j] = sign(sum);

    // Synchronise the threads
    __syncthreads();

    if (threadIdx.x == 0){
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
    }

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

    short int **d_array1;
    short int **d_array2;

    cudaMalloc((void**)&d_array1, sizeof(short int) * (N + 2));

    initializeArray(array1);
    printArray(array1);

    simulateIsing(array1, array2, 10);

    // free memory
    for (int i = 0; i < N + 2; i++){
        free(array1[i]);
        free(array2[i]);
    }

    free(array1);
    free(array2);
    
    return 0;
}