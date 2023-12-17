import PySimpleGUI as sg
from model_parser import HelperOFRL, ModelParser
import multiprocessing
import subprocess
import threading
import os

h = HelperOFRL()

class TrainGUI:
    def __init__(self):
        # Define the layout of the GUI
        self.layout = [
            [sg.Text("Select FAST File:")],
            [sg.Input(key="-FASTfile-"), sg.FileBrowse(initial_folder=h.get_file_path("../FAST_cfg"), file_types=(("FAST Files", "*.fst"),))],
            [sg.Text("Select RL File:")],
            [sg.Input(key="-RLfile-"), sg.FileBrowse(initial_folder=h.get_file_path("../model_hyperparams"), file_types=(("JSON Files", "*.json"),)),sg.Button("Open RL File")],
            [sg.Button("Run Training"), sg.Button("Exit")]
        ]

        # Create the window
        self.window = sg.Window("Training GUI", self.layout)
        self.event_loop()
        self.terminate()

    def terminate(self):
        self.window.close()

    def event_loop(self):
        # Event loop
        while True:
            event, values = self.window.read()
            if event == sg.WINDOW_CLOSED or event == "Exit":
                break
            elif event == "Open RL File":
                rl_file = values["-RLfile-"]
                #open rl_file in vscode
                if os.path.isfile(rl_file):
                    sp = subprocess.Popen(["code", str(rl_file)])
                    sp.wait()
                    print("Exit code:", sp.returncode)
                else:
                    print("File not found", rl_file)
            elif event == "Run Training":
                rl_file = values["-RLfile-"]
                fast_file = values["-FASTfile-"]
                launch_training_process(rl_file, fast_file)

                
def launch_training_process(rl_file, fast_file):
    p = multiprocessing.Process(target=run_training, args=(rl_file, fast_file))
    p.start()
        
def run_training(rl_file, fast_file):
    print("Training process started")
    mp = ModelParser(rl_file, fast_file)
    mp.learn()    

if __name__ == "__main__":
    try:
        TrainGUI()
    except Exception as e:
        print("An error occurred:", str(e))

