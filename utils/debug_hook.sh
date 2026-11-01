#!/bin/sh

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

GDB_HOST=$(hostname)
#printf "\n\t%sHOST: %s%s\n" "$RED" "$GDB_HOST" "$NC"

# Extract the binary name from the full path
BINARY_NAME=$(basename "$1")

if [ "$CCL_DEBUG_TEST" = "$BINARY_NAME" ]; then
    if [ "$PMI_RANK" -eq 0 ]; then        
        GDB_PORT=$((60000 + PMI_RANK))
        echo -e "GDB server for rank $PMI_RANK available on ${GREEN}$GDB_HOST:$GDB_PORT${NC}"
        exec gdbserver --no-startup-with-shell :$GDB_PORT "$@"
    else
        exec "$@"
    fi
else
    exec "$@"
fi
