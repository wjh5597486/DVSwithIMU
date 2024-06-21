import threading
import os
import time

def beep(sound):
    os.system(f'afplay /System/Library/Sounds/{sound}.aiff')

start_beep = threading.Thread(target=beep, args=("Tink",))
start_beep.daemon = True
end_beep = threading.Thread(target=beep, args=("Pop",))
end_beep.daemon = True

while True:
    start_beep.start()
    # 메인 코드 실행
    for i in range(5):
        print(i)
    end_beep.start()

print("Main code finished.")
