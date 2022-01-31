#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define N 10000
#define BLK_THREADS 32
#define SM 6
#define SM_CORES 128
#define ITERATIONS 50

/**
 * This is strongly hardware related. This specific value is calculated for 96KB of shared memory per multiprocessor.
 * 
 * This number ensures that for 128 cuda cores per multiprocessor (GTX 1050 Ti) and 32 threads per block if 4 blocks 
 * are running at the same time in the same multiprocessor the shared memory is enough for all of them. The actual number
 * is 12.288 but 12100 (110^2) is the closest perfect square. The actual value is 108 because of the 2 wrapping lines
 * on both sides of the array. 
 */
#define MAX_BLK_SIZE 70  // was 108


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
 * @brief Kernel function that simulates the ising model. Every thread calculates more than one moments using shared memory.
 * 
 * @param arraySize        The size of the global memory array (no wrappings)
 * @param sharedArraySize  The size of the shared memory array (WITH wrappings)
 * @param d_read           The global memory array to read from
 * @param d_write          The global memory array to write to
 * @param blocksPerRow     The number of blocks opend for one side of the array (total number of blocks is this values squared)
 * @param momentsPerThread The number of moments every thread will calculate
 */
__global__ void simulateIsing(int arraySize, int sharedArraySize, short int *d_read, short int *d_write, int blocksPerRow, int momentsPerThread){

    // The size of the shared memory is depended on the size of the sub array.
    __shared__ short int sharedArray[(MAX_BLK_SIZE + 2) * (MAX_BLK_SIZE + 2)];

    // i and j are the indexes of the first element of each sub block on the global array.
    int i = (blockIdx.x / blocksPerRow) * (sharedArraySize - 2);
    int j = (blockIdx.x % blocksPerRow) * (sharedArraySize - 2);


    // Acount for subarrays overlapping
    j = (j + sharedArraySize > arraySize) ? arraySize + 2 - sharedArraySize : j;

    i = (i + sharedArraySize > arraySize) ? arraySize + 2 - sharedArraySize : i;

    // Index for the flatted out 2D array. This index is the index of the firste moment of the block on the global memory
    int globalIndex = (arraySize + 2) * i + j;
    

    // Copy all the elements from the global memory to the shared memory
    for (int x = 0; x < sharedArraySize; x += blockDim.x){

        // Some of the threads at the end of the array go out of bounds.
        // This is because the number of the sub array is not always divided with the number of threads
        // The extra threads simply exit the loop and wait on the syncronise
        if (threadIdx.x + x < sharedArraySize){
            // The index to the shared memory
            int sharedOffset = (threadIdx.x + x) * sharedArraySize;

            // The index to the global memory
            int globalOffset = globalIndex + (threadIdx.x + x) * (arraySize + 2);


            // Every thread copies a row of momnets
            for (int k = 0; k < sharedArraySize; k++) {
                sharedArray[sharedOffset + k] = d_read[globalOffset + k];
            }
        }
    }

    // DEBUG prints
    // if (threadIdx.x == 0 && blockIdx.x == 0){
    //     printDeviceArray(d_read, arraySize);

    //     printDeviceArray(sharedArray, sharedArraySize - 2);
    // }


    // Threads need to be synced because all the data must be fetched from
    // the global memory before starting to caclulate the new array
    __syncthreads();


    int stride = blockDim.x;
    
    // From this point forward the problem to be solved is that of the small sub array. 
    // To convert this problem to something previously solve the wrapping lines are subtracted
    // from the size because they are acounted for on the formulas below
    sharedArraySize -= 2;

    for (int k = 0; k < blockDim.x * momentsPerThread; k = k + stride) {

        // This is the unique value that every thread in every block gets. 
        // This index does not take in to acount the wrapping lines (See initializeArray function for the wrapping system)
        int index_2d = threadIdx.x + k;

        /**
        * Index for the flatted out 2D array. The array passed in the GPU memory is 1D and already has the wrappings included.
        * To find the corespondance of the 2D index to the 1D index we do the following. The first size + 3 (size = N) elements of the array
        * are elements from the wrapping rows and columns. After that the index_2d is added and finaly index_2d / size calculates the line of the
        * element on the 2D array. For every line we need to add 2 because the first and last element of the array are there for wrapping
        */
        int index = sharedArraySize + 3 + index_2d + (index_2d / sharedArraySize) * 2;

        // if (blockIdx.x == 0){
        //         printf("blockId: %d, threadId: %d, index: %d, global index: %d\n", blockIdx.x, threadIdx.x, index, globalIndex);
        // }

        if (index <= (sharedArraySize + 1) * (sharedArraySize  + 2) - 2){
            
            // printf("block id %d, t_id %d , index %d\n", blockIdx.x, threadIdx.x, index);
            // if (blockIdx.x == 0){
            //     printf("blockId: %d, threadId: %d, index: %d, global index: %d\n", blockIdx.x, threadIdx.x, index, newIndex);
            // }


            // The sum of all the neighbors read from the sharedArray
            int sum = sharedArray[index - 1] + sharedArray[index + 1] + sharedArray[index - (sharedArraySize + 2)] + sharedArray[index + (sharedArraySize + 2)] + sharedArray[index];

            // The index that connects the local memory to the global memory
            int newIndex = globalIndex + arraySize + 3 + index_2d + (index_2d / sharedArraySize) * (2 + arraySize - sharedArraySize); 

            // Write the elemets on the global memory. There is no need to transfer the write array to the Shared memory
            // because only one write access is performed per moment
            d_write[newIndex] = sign(sum);

        } else {
            // If a thread goes out of bounds brakes from the loop
            break;
        }
    }

    // __syncthreads();
    // if (threadIdx.x == 0 && blockIdx.x == 0){
    //     printDeviceArray(sharedArray, sharedArraySize);
    //     printDeviceArray(sharedArrayWrite, sharedArraySize);
    // }

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
    
    __shared__ int shArr[BLK_THREADS];
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


/**
 * @brief Finds the optimal value for the sub block size for minimum overlapping
 * 
 * @param size   The array size
 * @param block  The targeted block size
 * @return       The best size for the block of moments (The one that produces less overlapping)
 */
int optimalBlockSize(int size, int block){
    int fit = block;
    
    int initialBlock = block;
    
    int optimalFit = fit;
    int optimalBlock = block;
    
    while (fit != 0 && block >= initialBlock - 20){
        
        fit = block - size % block;
        
        if (fit == block){
            fit = 0;
        }
        
        if (fit < optimalFit){
            optimalFit = fit;
            optimalBlock = block;
        }
        
        // printf("size: %d, block: %d, fit: %d\n", size, block, fit);
        
        block -= 1;
    }
    
    return optimalBlock;
}


int calculateGrid(int *nSimBlocks, int *momentsPerThread, int *blockSize, int arraySize){

    // This is the minimum number of blocks the GPU needs to be 100% utilized. The formula is (Cores per SM div Block size) * Number of SMs 
    int minBlocks = SM_CORES / BLK_THREADS * SM;
    int sharedArraySize;

    // This is the number of the maximum number of moments a block can have based on the amount of shared memory
    int maxMomentsPerBlock = MAX_BLK_SIZE * MAX_BLK_SIZE;

    // This is the minimum total number of moments needed to utilize fully both the shared memory and the CUDA cores of the GPU
    int minTotalMoments = minBlocks * maxMomentsPerBlock;

    // For arrays bigger that minTotalMoments we calculate the block size based on the MAX_BLK_SIZE. This ensures maximum shared memory efficiency
    // since we need to open more blocks than the GPU can run at once.
    if (arraySize * arraySize >= minTotalMoments){
        
        sharedArraySize = optimalBlockSize(arraySize, MAX_BLK_SIZE);  // The best block size for minimal overlap. This is the size without the wrapping lines

        // If the array size is not divided exactly with the block size the last blocks will overlap.
        // This will result on some moments being calculated twice but the number of them will be very small
        // compared with the total moments (less that 1.8%) given a big array.
        *nSimBlocks = arraySize % sharedArraySize == 0 ? arraySize / sharedArraySize : arraySize / sharedArraySize + 1;

        // The number of moments each thread will calculate. The number of threads per blocks is constant
        *momentsPerThread = (sharedArraySize * sharedArraySize) % BLK_THREADS == 0 ? (sharedArraySize * sharedArraySize) / BLK_THREADS : (sharedArraySize * sharedArraySize) / BLK_THREADS + 1;

        *blockSize = BLK_THREADS;

        sharedArraySize += 2;  // Add 2 to the sharedArraySize so that it contains the wrapping lines

    } else {
        // If the total number of moments is less than the minimum we will not need all of the shared memory.
        // In this case we will open the minimum number of blocks that can utilize the GPU fully in terms of cores
        
        // The targeted sharedArraySize
        sharedArraySize = arraySize % 8 == 0 ? arraySize / 8 : arraySize / 8 + 1;

        if (sharedArraySize >= 1){
        
            // If the array size is not divided exactly with the block size the last blocks will overlap.
            // This will result on some moments beign calculated twice but the number of them will be very small
            // compared with the total moments (less that 1.8%) given a big array.
            *nSimBlocks = 8;

            // The number of moments each thread will calculate. The number of threads per blocks is constant
            *momentsPerThread = (sharedArraySize * sharedArraySize) % BLK_THREADS == 0 ? (sharedArraySize * sharedArraySize) / BLK_THREADS : (sharedArraySize * sharedArraySize) / BLK_THREADS + 1;

            
            *blockSize = sharedArraySize * sharedArraySize;
        }

        sharedArraySize += 2;  // Add 2 to the sharedArraySize so that it contains the wrapping lines
    }

    return sharedArraySize;
}


int main(int argc, char **argv){

    // The array is N + 2 size for the wrapping around on both dimensions.
    short int **array1 = (short int **) calloc ((N + 2), sizeof(short int *));

    for (int i = 0; i < N + 2; i++){
        array1[i] = (short int *) calloc ((N + 2), sizeof(short int));
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
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("Host -> Device time:  %3.1f ms \n\n", time);

    cudaEventRecord(start, 0);

    int stabilityBlocks = (N * N % BLK_THREADS) ? (N * N / BLK_THREADS + 1) : N * N / BLK_THREADS;

    // Unified memory pointer for detecting stable state
    int *stable_state;
    cudaMallocManaged((void **) &stable_state, 3 * sizeof(int));  // Allocate pointer for device and host access (unified memory)

    // Initialize the stable state array with INT_MAX
    stable_state[0] = INT_MAX;
    stable_state[1] = INT_MAX - 1;
    stable_state[2] = INT_MAX - 2;

    short int* dev_out;
    cudaMallocManaged((void **)&dev_out, sizeof(short int) * stabilityBlocks);

    int nSimBlocks;
    int momentsPerThread;
    int blockSize;
    
    int sharedArraySize = calculateGrid(&nSimBlocks, &momentsPerThread, &blockSize, N);

    printf("Number of blocks: %d, moments per thread: %d, shared array size: %d, block size %d\n\n", nSimBlocks * nSimBlocks, momentsPerThread, sharedArraySize, blockSize);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("Memory allocations time:  %3.1f ms \n\n", time);
    
    cudaEventRecord(start, 0);

    for (int iteration = 0; iteration < ITERATIONS; iteration++){

        // Call the kernel with numberOfBlocks blocks and N_threads. This call introduces a restriction on the size of the array
        // The max number of threads per block is 1024 so the max N is theoretically 32 (practically 30 because of the wrappings)
        simulateIsing <<< nSimBlocks * nSimBlocks, blockSize >>> (N,  sharedArraySize, d_array1, d_array2, nSimBlocks, momentsPerThread);  //sharedArraySize * sharedArraySize * sizeof(short int)

        cudaDeviceSynchronize();

        int wrappingBlocks = (N % BLK_THREADS) ? (N / BLK_THREADS + 1) : N / BLK_THREADS;

        completeWrapping <<<wrappingBlocks, BLK_THREADS>>> (d_array2, N);
        cudaDeviceSynchronize();

        // UNCOMMENT for debug prints
        // debugPrints <<<1, 1>>> (d_array2, N);
        // cudaDeviceSynchronize();
        


        // UCOMMENT to stop the sim once a stable state is reached

        // detectStableState <<<stabilityBlocks, BLK_THREADS>>> (dev_out, d_array2, (N + 2) * (N + 2));

        // detectStableState <<<1, BLK_THREADS>>> (dev_out, dev_out, stabilityBlocks);
        // cudaDeviceSynchronize();

        // stable_state[iteration % 3] = dev_out[0];

        // printf("Iteration: %d, energy %d\n\n\n", iteration, stable_state[iteration % 3]);

        // if (stable_state[0] == stable_state[2]) {
        //     printf("\n\n C O N V E R G E N C E !! @ iteration %d\n\n", iteration);
        //     break;
        // }

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
        // cudaMemcpy(array2[i], d_array2 + i * (N + 2), sizeof(short int) * (N + 2), cudaMemcpyDeviceToHost);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("MEmory Device -> Host time:  %3.1f ms \n\n", time);

    // free memory
    for (int i = 0; i < N + 2; i++){
        free(array1[i]);
    }

    free(array1);
    
    cudaFree(d_array1);
    cudaFree(d_array2);

    return 0;
}


