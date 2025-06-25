#!/bin/bash
# Shell script to easily list database users through Docker Compose
# This script must be run from the logios directory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if we're in the right directory
check_directory() {
    if [[ ! -f "docker-compose.yaml" ]]; then
        print_error "docker-compose.yaml not found. Please run this script from the logios directory."
        exit 1
    fi
}

# Function to check if Docker Compose is available
check_docker_compose() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if command -v docker-compose &> /dev/null; then
        DOCKER_COMPOSE_CMD="docker-compose"
    elif docker compose version &> /dev/null; then
        DOCKER_COMPOSE_CMD="docker compose"
    else
        print_error "Docker Compose is not available"
        exit 1
    fi
}

# Function to check if the flaskui container is running
check_container() {
    if ! $DOCKER_COMPOSE_CMD ps flaskui | grep -q "Up"; then
        print_warning "flaskui container is not running. Starting services..."
        $DOCKER_COMPOSE_CMD up -d flaskui
        sleep 5
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  list          List all users with full details (default)"
    echo "  simple        List users with basic info only"
    echo "  admins        List only admin users"
    echo "  active        List only active users"
    echo "  inactive      List only inactive users"
    echo "  stats         Show user statistics"
    echo "  help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                 # List all users"
    echo "  $0 simple          # List users with basic info"
    echo "  $0 admins          # List admin users only"
    echo "  $0 stats           # Show user statistics"
    echo ""
    echo "Note: This script must be run from the logios directory."
}

# Main function
main() {
    local command=${1:-list}
    
    case $command in
        list)
            print_info "Listing all users with full details..."
            $DOCKER_COMPOSE_CMD exec flaskui python user_manager.py list
            ;;
        simple)
            print_info "Listing users with basic info..."
            $DOCKER_COMPOSE_CMD exec flaskui python user_manager.py list-simple
            ;;
        admins)
            print_info "Listing admin users..."
            $DOCKER_COMPOSE_CMD exec flaskui python user_manager.py list-admins
            ;;
        active)
            print_info "Listing active users..."
            $DOCKER_COMPOSE_CMD exec flaskui python user_manager.py list-active
            ;;
        inactive)
            print_info "Listing inactive users..."
            $DOCKER_COMPOSE_CMD exec flaskui python user_manager.py list-inactive
            ;;
        stats)
            print_info "Showing user statistics..."
            $DOCKER_COMPOSE_CMD exec flaskui python user_manager.py stats
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            print_error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

# Pre-flight checks
check_directory
check_docker_compose
check_container

# Run main function
main "$@"

print_success "Command completed successfully!"
