import os
import subprocess
import time

# Configuration
aria2c_path = "aria2c"  # Ensure aria2c is installed and in PATH
download_link = "https://genome-idx.s3.amazonaws.com/kraken/k2_core_nt_20240904.tar.gz"
retry_delay = 5  # Delay between retries in seconds

# Get the current directory to save the file
download_dir = os.getcwd()

# Function to download a file with infinite retries
def download_file(link):
    attempt = 1
    while True:
        try:
            # Construct the aria2c command
            command = [
                aria2c_path,
                "--dir", download_dir,
                "--max-connection-per-server=10",  # Increased connections for better speed
                "--split=10",
                "--min-split-size=1M",
                "--retry-wait=5",
                "--max-tries=0",  # Infinite retries within aria2c
                link
            ]
            print(f"Attempt {attempt}: Downloading {link}...")
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                print(f"Download successful: {link}")
                break
            else:
                print(f"Download failed. Retrying in {retry_delay} seconds...")
        except Exception as e:
            print(f"Error occurred: {e}")

        attempt += 1
        time.sleep(retry_delay)

# Main function
def main():
    print(f"Starting download for {download_link} into directory: {download_dir}")
    download_file(download_link)
    print("Download completed successfully!")

if __name__ == "__main__":
    main()
