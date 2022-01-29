#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <limits.h>
#include <cuda.h>

#define N 20000  // The size of the initial array (N * N moments)
#define N_THREADS 1024  // Number of threads per block
#define N_ITER 10  // The number of iterations

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

            int rnd = rand() % 1000;  // Get a double random number in (0,1) range
            
            // 0.5 is chosen so that +1 and -1 are 50% each
            if (rnd >= 500){

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
__global__ void simulateIsing(short int *d_read, short int *d_write, int size){

    // This is the unique value that every thread in every block gets. This value represents the index of the NxN array
    // of the moment to ble calculated. This index does not take in to acount the wrapping lines (See initializeArray function for the wrapping system)
    int index_2d = blockIdx.x * blockDim.x + threadIdx.x;

    /**
    * Index for the flatted out 2D array. The array passed in the GPU memory is 1D and already has the wrappings included.
    * To find the corespondance of the 2D index to the 1D index we do the following. The first size + 3 (size = N) elements of the array
    * are elements from the wrapping rows and columns. After that the index_2d is added and finaly index_2d / size calculates the line of the
    * element on the 2D array. For every line we need to add 2 because the first and last element of the array are there for wrapping
    */
    int index = size + 3 + index_2d + (index_2d / size) * 2;  

    // If the total number of threads is enough for all the N * N elements of the array, calculate their new moments.
    if (index <= (size + 1) * (size  + 2) - 2){
        // printf("block id %d, t_id %d , index %d\n", blockIdx.x, threadIdx.x, index);

        // Sum all the neighbours. The index - (size + 2) is the above element index + (size + 2) is the below element. The 2 factor is added 
        // Because the array with the wrappings is 2 rows bigger that the initial row
        int sum = d_read[index - 1] + d_read[index + 1] + d_read[index - (size + 2)] + d_read[index + (size + 2)] + d_read[index];

        d_write[index] = sign(sum);  // Update the value of this moment
    }
    
}


/**
 * @brief This function calculates and updates the outer rows and columns that contain the oposite side elements so that the wrapping 
 * can be performed.
 * 
 * @param d_write The just written array to update the wraps
 * @param size The size of the initial array (without the wraps)
 */
__global__ void completeWrapping(short int *d_write, int size){
    
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if(j <= size){

        // Update the wrapping rows...
        d_write[size * (size + 2) + size + 2 + j] = d_write[size + 2 + j];  // This formula transforms 2D coordinates to 1D
        d_write[size * (size + 2) + j - size * (size + 2)] = d_write[size * (size + 2) + j];  // This formula transforms 2D coordinates to 1D
        
        
        // ... and columns as well
        d_write[j * (size + 2) + 1 + size] = d_write[j * (size + 2) + 1];  // This formula transforms 2D coordinates to 1D
        d_write[j * (size + 2)] = d_write[j * (size + 2) + size];  // This formula transforms 2D coordinates to 1D
    }
}

__global__ void debugPrints(short int *arr, int size){
    printDeviceArray(arr, size);
}


/**
 * @brief Sums in parallel all the elements of the array. To perform the operation the function must be called twice
 * The first time the array is spit in blocks and the sum of every block is calculated. The sum of every block is then returned
 * in the dout array. The second time of the function is called with the dout as input. The sum of the dout array is returned in the 
 * dout[0]
 * 
 * @param d_out The array to return the results
 * @param arr The input array
 * @param arr_size The size of the inputu array
 */
__global__ void detectStableState(short int *d_out, short int *arr, int arr_size){
    int thIdx = threadIdx.x;
    int gthIdx = thIdx + blockIdx.x * blockDim.x;

    const int gridSize = blockDim.x * gridDim.x;
    
    int sum = 0;
    
    for (int i = gthIdx; i < arr_size; i += gridSize)
        sum += arr[i];
    
    __shared__ int shArr[N_THREADS];
    shArr[thIdx] = sum;

    __syncthreads();
    
    
    for (int size = blockDim.x / 2; size > 0; size /= 2) { //uniform
        if (thIdx<size)
            shArr[thIdx] += shArr[thIdx+size];

        __syncthreads();
    }

    if (thIdx == 0)
        d_out[blockIdx.x] = shArr[0];
    
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

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    printf("\nStarting simulation \n\n");

    cudaEventRecord(start, 0);

    // Allocate the memory for the device arrays
    cudaMalloc((void**)&d_array1, sizeof(short int) * (N + 2) * (N + 2));
    cudaMalloc((void**)&d_array2, sizeof(short int) * (N + 2) * (N + 2));


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("Memory allocations time:  %3.1f ms \n\n", time);

    cudaEventRecord(start, 0);

    // Initialize the array 1 with random -1 and 1 values (50% distribution)
    initializeArray(array1);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("Initialization time:  %3.1f ms \n\n", time);

    cudaEventRecord(start, 0);


    // Copy the host memory to the device memory. This transfer also converts the host 2D array to 1D for the device
    for (int i = 0; i < N + 2; i++) {
        cudaMemcpy(d_array1 + i * (N + 2), array1[i], sizeof(short int) * (N + 2), cudaMemcpyHostToDevice);
        // cudaMemcpy(d_array2 + i * (N + 2), array2[i], sizeof(short int) * (N + 2), cudaMemcpyHostToDevice);
    }
    

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("Host -> Device time:  %3.1f ms \n\n", time);

    cudaEventRecord(start, 0);

    int numberOfBlocks;

    if((N * N) % N_THREADS == 0){
        numberOfBlocks = (N * N) / N_THREADS;
    }
    else{ 
        numberOfBlocks = (N * N) / N_THREADS + 1;
    }

    // Unified memory pointer for detecting stable state
    int *stable_state;
    cudaMallocManaged((void **) &stable_state, 3 * sizeof(int));  // Allocate pointer for device and host access (unified memory)

    // Initialize the stable state array with INT_MAX
    stable_state[0] = INT_MAX;
    stable_state[1] = INT_MAX - 1;
    stable_state[2] = INT_MAX - 2;

    short int* dev_out;
    cudaMallocManaged((void **)&dev_out, sizeof(short int) * numberOfBlocks);

    cudaEventRecord(start, 0);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("Memory allocations time:  %3.1f ms \n\n", time);
    
    cudaEventRecord(start, 0);

    for (int iteration = 0; iteration < N_ITER; iteration++){
        
        // Call the kernel with 1 block and N^2 threads. This call introduces a restriction on the size of the array
        // The max number of threads per block is 1024 so the max N is theoreticaly 32 (practicaly 30 because of the wrappings)
        simulateIsing <<<numberOfBlocks, N_THREADS>>> (d_array1, d_array2, N);
        cudaDeviceSynchronize();

        numberOfBlocks = (N % N_THREADS) ? (N / N_THREADS + 1) : N / N_THREADS;

        // Complete the wrapping lines for the array that was just calculated
        completeWrapping <<<numberOfBlocks, N_THREADS>>> (d_array2, N);
        cudaDeviceSynchronize();

        // UNCOMMENT for debug prints
        
        /*
        debugPrints <<<1, 1>>> (d_array2, N);
        cudaDeviceSynchronize();

        debugPrints <<<1, 1>>> (d_array1, N);
        cudaDeviceSynchronize();
        */


        // UCOMMNET to stop the sim once a stable state is reached
        /*
        numberOfBlocks = (N * N % N_THREADS) ? (N * N / N_THREADS + 1) : N * N / N_THREADS;
        
        detectStableState <<<numberOfBlocks, N_THREADS>>> (dev_out, d_array2, (N + 2) * (N + 2));

        detectStableState <<<1, N_THREADS>>> (dev_out, dev_out, numberOfBlocks);
        cudaDeviceSynchronize();

        stable_state[iteration % 3] = dev_out[0];

        // printf("Iteration: %d, energy %d\n\n\n", iteration, stable_state[iteration % 3]);

        if (stable_state[0] == stable_state[2]) {
            printf("\n\n C O N V E R G E N C E !! @ iteration %d\n\n", iteration);
            break;
        }
        */
        
        // Swap the two arrays.
        short int *tmp = d_array2;
        d_array2 = d_array1;
        d_array1 = tmp;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("GPU run time:  %3.1f ms \n\n", time);


    cudaEventRecord(start, 0);

    // Copy the device memory back to host again converting from 1D device array to 2D host array
    for (int i = 0; i < N + 2; i++) {
        cudaMemcpy(array1[i], d_array1 + i * (N + 2), sizeof(short int) * (N + 2), cudaMemcpyDeviceToHost);
        //cudaMemcpy(array2[i], d_array2 + i * (N + 2), sizeof(short int) * (N + 2), cudaMemcpyDeviceToHost);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("MEmory Device -> Host time:  %3.1f ms \n\n", time);

    // free memory
    for (int i = 0; i < N + 2; i++){
        free(array1[i]);
        free(array2[i]);
    }

    free(array1);
    free(array2);
    
    cudaFree(d_array1);
    cudaFree(d_array2);
    cudaFree(dev_out);
    cudaFree(stable_state);

    return 0;
}