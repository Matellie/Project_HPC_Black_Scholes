# Compiler and flags
CXX := g++
CXXFLAGS := -O3 -fopenmp -Wall -Wextra -pedantic
CUDA_COMPILER := nvcc
CUDA_FLAGS := -O3 -Xcompiler -fopenmp

# Directories
SRC_DIR := src
BUILD_DIR := build

# Find all source files
CPP_SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
CU_SOURCES := $(wildcard $(SRC_DIR)/*.cu)

# Create corresponding object files
CPP_OBJECTS := $(CPP_SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
CU_OBJECTS := $(CU_SOURCES:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)

# Executables
CPP_EXECUTABLES := $(CPP_SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%)
CU_EXECUTABLES := $(CU_SOURCES:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%)

# Default target
all: $(CPP_EXECUTABLES) $(CU_EXECUTABLES)

# Rule to build C++ executables
$(BUILD_DIR)/%: $(BUILD_DIR)/%.o
	$(CXX) $(CXXFLAGS) -o $@ $<

# Rule to compile C++ source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Rule to build CUDA executables
$(BUILD_DIR)/%: $(BUILD_DIR)/%.cu.o
	$(CUDA_COMPILER) $(CUDA_FLAGS) -o $@ $<

# Rule to compile CUDA source files
$(BUILD_DIR)/%.cu.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(CUDA_COMPILER) $(CUDA_FLAGS) -c -o $@ $<

# Create build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean up build artifacts
clean:
	rm -rf $(BUILD_DIR)

# Phony targets
.PHONY: all clean
