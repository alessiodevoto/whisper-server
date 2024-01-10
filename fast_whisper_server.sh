#!/bin/bash
# This script is used to monitor the server for crashes and restart it if necessary
server_running=1 

# Set the file to monitor
# When there is a change in this file, we have to shut down and restart the server
file_to_monitor="/workspace/server_crashes_log.txt"

# Set the interval for checking changes in seconds
check_interval=5

# Function to check for file changes and restart the server if necessary
check_and_restart_server() {
    current_checksum=$(md5sum "$file_to_monitor")

    if [ "$current_checksum" != "$last_checksum" ]; then
        echo "File has changed. Restarting server..."
        kill_server
        $start_server_command &
        last_checksum="$current_checksum"
    fi
}


# Function to kill the server process
kill_server() {
    # this is equivalent to pgrep -f "$start_server_command"
    server_process_pid=$(ps aux | grep "$start_server_command" | grep -v grep | awk '{print $2}')

    # Server might have been killed already
    if [ -z "$server_process_pid" ]; then
        echo "Server is not running anymore."
        return
    fi
    
    # Attempt to kill the server process
    echo -e "\nGracefully killing server process with PID: $server_process_pid"
    kill "$server_process_pid"
    if [ $? -eq 0 ]; then
        echo "Kill successful!"
    else
        echo "Kill failed. Server shutting down."
        exit 1
    fi
}


# Function to kill the servr process and exit the script
kill_server_and_exit() {
    kill_server
    exit 0
}

trap kill_server_and_exit SIGINT SIGTERM EXIT


# Initial checksum
last_checksum=$(md5sum "$file_to_monitor")

# Get the command to start the server from the command line arguments
echo "Starting Whisper server..."
start_server_command="python3 /workspace/whisper-server/fast_whisper_app.py $@"
$start_server_command &

# Main loop
while true; do
    sleep ${check_interval}
    check_and_restart_server
done


