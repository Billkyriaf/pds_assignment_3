CC = gcc
NVCC = nvcc

BUILD_DIR := ./build
SRC_DIRS := ./src

# Colors
# GREEN = \033[1;32m
# RED = \033[1;31m
# NC = \033[0m

GREEN = 
RED = 
NC = 

CFLAGS := -O3


sequential: pre_build build_sequential post_build

pre_build:
	@echo "$(GREEN)Building all...$(NC)"


post_build:
	@echo "$(GREEN)Build all successful!!$(NC)"
	@echo
	@echo


######## V0 sequential  ########

build_sequential:
	@echo "    $(GREEN)Building Serial binary...$(NC)"
	@$(CC) $(CFLAGS) -o $(BUILD_DIR)/sequential.out $(SRC_DIRS)/sequential.c
	@echo "    $(GREEN)Build finished successfully!$(NC)"
	@echo

run_sequential_hpc: sequential
	@rm -f *.out         # remove the previous slurm-*.out file 
	@sbatch deploy.sh    # queue the next job
	@sleep 2             # wait for 2 seconds for the new slurm-*.out file to be created
	@echo
	@echo
	@timeout 10 tail -fn +1 *.out || exit 0  # tail the new slurm-*.out file for 5 seconds before exiting with code 0

run_sequential: sequential
	@$(BUILD_DIR)/sequential.out


######## V1 One moment per thread and singl3 block  ########

single_moment: pre_build build_single_moment post_build

build_single_moment:
	@mkdir -p $(BUILD_DIR)
	@echo "    $(GREEN)Building single block CUDA binary...$(NC)"
	@$(NVCC) -o $(BUILD_DIR)/single_moment.out $(SRC_DIRS)/GPU_single_moment.cu
	@echo "    $(GREEN)Build finished successfully!$(NC)"
	@echo

run_single_moment_hpc: single_moment
	@rm -f *.out                         # remove the previous slurm-*.out file 
	@sbatch hpc_deploy_single_moment.sh  # queue the next job
	@sleep 2                             # wait for 2 seconds for the new slurm-*.out file to be created
	@echo
	@echo
	@timeout 10 tail -fn +1 *.out || exit 0  # tail the new slurm-*.out file for 5 seconds before exiting with code 0

run_single_moment: single_moment
	@echo
	@echo
	@nvprof $(BUILD_DIR)/single_moment.out


######## V2 Block of moments per thread ########
block_moments: pre_build build_block_moments post_build

build_block_moments:
	@mkdir -p $(BUILD_DIR)
	@echo "    $(GREEN)Building blocks CUDA binary...$(NC)"
	@$(NVCC) -o $(BUILD_DIR)/block_moments.out $(SRC_DIRS)/GPU_block_moments.cu
	@echo "    $(GREEN)Build finished successfully!$(NC)"
	@echo	

run_block_moments_hpc: single_moment
	@rm -f *.out                         # remove the previous slurm-*.out file 
	@sbatch hpc_deploy_block_moments.sh  # queue the next job
	@sleep 2                             # wait for 2 seconds for the new slurm-*.out file to be created
	@echo
	@echo
	@timeout 10 tail -fn +1 *.out || exit 0  # tail the new slurm-*.out file for 5 seconds before exiting with code 0

run_block_moments: block_moments
	@echo
	@echo
	@nvprof $(BUILD_DIR)/block_moments.out


######## V3 Shared memory ########
shared_memory: pre_build build_shared_memory post_build

build_shared_memory:
	@mkdir -p $(BUILD_DIR)
	@echo "    $(GREEN)Building blocks CUDA binary...$(NC)"
	@$(NVCC) -o $(BUILD_DIR)/shared_memory.out $(SRC_DIRS)/GPU_shared_mem.cu
	@echo "    $(GREEN)Build finished successfully!$(NC)"
	@echo	

run_shared_memory: build_shared_memory
	@echo
	@echo
	@nvprof ./build/shared_memory.out	

run_shared_memory_hpc: single_moment
	@rm -f *.out                  # remove the previous slurm-*.out file 
	@sbatch hpc_shared_memory.sh  # queue the next job
	@sleep 2                      # wait for 2 seconds for the new slurm-*.out file to be created
	@echo
	@echo
	@timeout 10 tail -fn +1 *.out || exit 0  # tail the new slurm-*.out file for 5 seconds before exiting with code 0


.PHONY: clean
clean:
	@echo "$(RED)Clearing build directory...$(NC)"
	@rm -rf $(BUILD_DIR)
	@echo "$(GREEN)Done!$(NC)"