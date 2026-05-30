import cv2

def test_cameras():
    for idx in range(3):
        print(f"\n--- Probando cámara {idx} ---")
        
        # Test DSHOW
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            print(f"DSHOW: opened=True, read()={ret}")
            cap.release()
        else:
            print("DSHOW: opened=False")
            
        # Test MSMF
        cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
        if cap.isOpened():
            ret, frame = cap.read()
            print(f"MSMF: opened=True, read()={ret}")
            cap.release()
        else:
            print("MSMF: opened=False")
            
        # Test DEFAULT
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            print(f"DEFAULT: opened=True, read()={ret}")
            cap.release()
        else:
            print("DEFAULT: opened=False")

if __name__ == "__main__":
    test_cameras()
