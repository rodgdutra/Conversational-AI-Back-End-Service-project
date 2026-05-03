#!/bin/bash
# Script to run tests in the Docker environment

# Ensure script is executable
# chmod +x run-tests.sh

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Running tests in Docker environment ===${NC}"

# Check if docker-compose is available
if ! command -v docker compose &> /dev/null; then
    echo -e "${RED}Error: docker compose is not installed${NC}"
    exit 1
fi

# Parse command line arguments
TEST_ARGS=""
NO_REBUILD=false

for arg in "$@"; do
    case $arg in
        --no-rebuild)
            NO_REBUILD=true
            shift
            ;;
        *)
            TEST_ARGS="$TEST_ARGS $arg"
            ;;
    esac
done

# Set pytest arguments if provided
if [ -n "$TEST_ARGS" ]; then
    export PYTEST_ARGS="$TEST_ARGS"
    echo -e "${YELLOW}Running with arguments: ${PYTEST_ARGS}${NC}"
fi

# Cleanup old containers if they exist
echo -e "${GREEN}Cleaning up old test containers...${NC}"
docker compose -f docker/docker-compose.test.yml down

if [ "$NO_REBUILD" = false ]; then
    echo -e "${GREEN}Building test containers...${NC}"
    docker compose -f docker/docker-compose.test.yml build
fi

# Run the tests
echo -e "${GREEN}Starting tests...${NC}"
docker compose -f docker/docker-compose.test.yml up --abort-on-container-exit

# Cleanup when done
echo -e "${GREEN}Tests completed, cleaning up...${NC}"
docker compose -f docker/docker-compose.test.yml down

echo -e "${GREEN}Done!${NC}"
