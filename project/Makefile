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

# Directories
# LIBRARIES_SRC := $(shell find $(SRC_DIRS)/libraries $(SRC_DIRS)/ext_libraries -name '*.c')
# LIBRARIES_SRC := $(LIBRARIES_SRC:%=$(BUILD_DIR)/%.o)

SEQUENTIAL_SRC := $(shell find $(SRC_DIRS) -name '*.c') # $(SRC_DIRS)/sequential.c
SEQUENTIAL_SRC := $(SEQUENTIAL_SRC:%=$(BUILD_DIR)/%.o)


# Every folder in ./src will need to be passed to GCC so that it can find header files
INC_DIRS := $(shell find $(SRC_DIRS) -type d)
# Add a prefix to INC_DIRS. So moduleA would become -ImoduleA. GCC understands this -I flag
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CFLAGS := $(INC_FLAGS) -O3


sequential: pre_build build_sequential post_build

pre_build:
	@echo "$(GREEN)Building all...$(NC)"


post_build:
	@echo "$(GREEN)Build all successful!!$(NC)"
	@echo
	@echo


build_sequential: $(SEQUENTIAL_SRC)
	@echo "    $(GREEN)Building Serial binary...$(NC)"
	@$(CC) $(CFLAGS) -o $(BUILD_DIR)/sequential.out $(SEQUENTIAL_SRC)
	@echo "    $(GREEN)Build finished successfully!$(NC)"
	@echo

run_sequential: sequential
	@rm -f *.out         # remove the previous slurm-*.out file 
	@sbatch deploy.sh    # queue the next job
	@sleep 2             # wait for 2 seconds for the new slurm-*.out file to be created
	@echo
	@echo
	@timeout 5 tail -fn +1 *.out || exit 0  # tail the new slurm-*.out file for 5 seconds before exiting with code 0


$(BUILD_DIR)/%.c.o: %.c
	@mkdir -p $(dir $@)
	@$(CC) $(CFLAGS) -c $< -o $@


######## V1 One moment per thread and singl3 block  ########

single_block: pre_build build_single_block post_build


build_single_block:
	@mkdir -p $(BUILD_DIR)
	@echo "    $(GREEN)Building single block CUDA binary...$(NC)"
	@$(NVCC) -o $(BUILD_DIR)/single_block.out $(SRC_DIRS)/gpu_1t_per_moment.cu
	@echo "    $(GREEN)Build finished successfully!$(NC)"
	@echo

run_single_block: single_block
	@rm -f *.out                     # remove the previous slurm-*.out file 
	@sbatch deploy_single_block.sh   # queue the next job
	@sleep 2                         # wait for 2 seconds for the new slurm-*.out file to be created
	@echo
	@echo
	@timeout 5 tail -fn +1 *.out || exit 0  # tail the new slurm-*.out file for 5 seconds before exiting with code 0

.PHONY: clean
clean:
	@echo "$(RED)Clearing build directory...$(NC)"
	@rm -rf $(BUILD_DIR)
	@echo "$(GREEN)Done!$(NC)"