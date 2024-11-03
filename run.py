import subprocess
import time


def main():
    # Start FastAPI server
    # api_process = subprocess.Popen(["python", "main.py"])

    # Start Streamlit server
    streamlit_process = subprocess.Popen(["streamlit", "run", "main.py"])

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # api_process.terminate()
        streamlit_process.terminate()


if __name__ == "__main__":
    main()
