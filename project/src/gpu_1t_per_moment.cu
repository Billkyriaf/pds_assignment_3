#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <limits.h>
#include <cuda.h>

#define N 30

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

/**
 * Prints an 1D array like a 2D array for every size + 2 elements
 * 
 * @param arr   The array to print
 * @param size  The size of the single row without wrapping
 */
__device__ void printDeviceArray(short int *arr, int size){

    for (int i = 0; i < size + 2; i++) {
        for(int j = 0; j < size + 2; j++){

            if (arr[i * (size + 2) + j] < 0){
                printf("%hd ", arr[i * (size + 2) + j]);
        
            } else {
                printf(" %hd ", arr[i * (size + 2) + j]);
            }
        }
        printf("\n");
    }
    printf("\n");
}

/**
 * Sign function implementation
 * 
 * @param sum The sum to find the sign
 * 
 * @return 1 if the sun is greater than 0 else -1
 * 
 */
__device__ short int sign(short int sum){
    return sum > 0 ? 1 : -1;
}


/**
 * Finds the sum of all the elements of an 1D array
 * 
 * @param arr   The array to find the sum
 * @param size  The size of the single row of the 2D array without wrapping
 * 
 * @return The sum of all the elements of the array
 */ 
__device__ int summation(short int *arr, int size){
    
    int sum = 0;
    
    for (int i = 0; i < (size + 2) * (size + 2); i++){
        sum += arr[i];
    }

    return sum;
}


/**
 * Kernel function that simulates the ising model. The function runs for a number of iterations and breaks if a stable state is reached.
 * 
 * @param d_read       The initial array
 * @param d_write      A copy of the initial array in which data are written to
 * @param size         The size of the single row of the 2D array without wrapping
 * @param iterations   The maximum number of iterations to run
 */ 
__global__ void simulateIsing(short int *d_read, short int *d_write, int size, int iterations){
    
    // if (threadIdx.x == 0) {
    //     printDeviceArray(d_read, size);
    // }
    
    // This array is shared among all the threads and is used for detecting steady state 
    __shared__ int previous_sum[3];
    
    // The first thread initializes the array
    if (threadIdx.x == 0) {
        previous_sum[0] = summation(d_read, size);  // Initial sum
        previous_sum[1] = INT_MAX - 1;  // Random value for initializing
        previous_sum[2] = INT_MAX - 2;  // Random value for initializing
    }
    
    // Run the simulation....
    for(int iteration = 0; iteration < iterations; iteration++){

        // Index for the flatted out 2D array. This formula is explained in the report.
        int index = size + 3 + threadIdx.x + (threadIdx.x / size) * 2;

        // Calculate the new spin for each point
        int sum = d_read[index - 1] + d_read[index + 1] + d_read[index - (size + 2)] + d_read[index + (size + 2)] + d_read[index];

        d_write[index] = sign(sum);  // Update the value of this moment

        // Synchronise the threads here because the wrappings need the complete array
        __syncthreads();

        if(threadIdx.x == 0){
            // Update the wrapping rows...
            for (int j = 1; j <= N; j++){
                d_write[size * (size + 2) + size + 2 + j] = d_write[size + 2 + j];  // This formula transforms 2D coordinates to 1D
                d_write[size * (size + 2) + j - size * (size + 2)] = d_write[size * (size + 2) + j];  // This formula transforms 2D coordinates to 1D
            }
        }
        
        if(threadIdx.x == size){
            // ... and columns as well
            for (int i = 1; i <= N; i++){
                d_write[i * (size + 2) + 1 + size] = d_write[i * (size + 2) + 1];  // This formula transforms 2D coordinates to 1D
                d_write[i * (size + 2)] = d_write[i * (size + 2) + size];  // This formula transforms 2D coordinates to 1D
            }
        }

        // Synchronise the threads here because the summation function needs the complete array
        __syncthreads();

        if (threadIdx.x == 0) {
            // Find the sum and save it to the iteration % 3 index. This index takes the values 0, 1, 2, 0, 1, 2 ....
            previous_sum[iteration % 3] = summation(d_write, size);

            printf("Iteration: %d, energy %d\n", iteration, previous_sum[iteration % 3]);
        }


        // Synchronise threads here because all threads need the previous_sum values calculated above
        __syncthreads();


        // If two sums that are 1 iteration apart are equal, a steady state has been reached
        if (previous_sum[0] == previous_sum[2]) {
            break;
        }


        // Swap the two arrays.
        short int *tmp = d_write;
        d_write = d_read;
        d_read = tmp;
    }   
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

    // Device memory pointers
    short int *d_array1;
    short int *d_array2;


    // Allocate the memory for the device arrays
    cudaMalloc((void**)&d_array1, sizeof(short int) * (N + 2) * (N + 2));
    cudaMalloc((void**)&d_array2, sizeof(short int) * (N + 2) * (N + 2));

    // Initialize the array 1 with random -1 and 1 values (50% distribution)
    initializeArray(array1);


    // Copy the host memory to the device memory. This transfer also converts the host 2D array to 1D for the device
    for (int i = 0; i < N + 2; i++) {
        cudaMemcpy(d_array1 + i * (N + 2), array1[i], sizeof(short int) * (N + 2), cudaMemcpyHostToDevice);
        cudaMemcpy(d_array2 + i * (N + 2), array2[i], sizeof(short int) * (N + 2), cudaMemcpyHostToDevice);
    }
    
    // Call the kernel with 1 block and N^2 threads. This call introduces a restriction on the size of the array
    // The max number of threads per block is 1024 so the max N is theoreticaly 32 (practicaly 30 because of the wrappings)
    simulateIsing <<<1, N * N>>> (d_array1, d_array2, N, 500000);


    // Copy the device memory back to host again converting from 1D device array to 2D host array
    for (int i = 0; i < N + 2; i++) {
        cudaMemcpy(array1[i], d_array1 + i * (N + 2), sizeof(short int) * (N + 2), cudaMemcpyDeviceToHost);
        cudaMemcpy(array2[i], d_array2 + i * (N + 2), sizeof(short int) * (N + 2), cudaMemcpyDeviceToHost);
    }

    

    // free memory
    for (int i = 0; i < N + 2; i++){
        free(array1[i]);
        free(array2[i]);
    }

    free(array1);
    free(array2);
    
    cudaFree(d_array1);
    cudaFree(d_array2);

    return 0;
}