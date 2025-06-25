import ctypes
import numpy as np
import cv2

# Load the SecuGen library
dll_path = "C:/Program Files/SecuGen/FDx SDK Pro for Windows v4.3.1/bin/x64/sgfplib.dll"
sgfplib = ctypes.WinDLL(dll_path)

# Define return type for functions
sgfplib.Create.argtypes = []
sgfplib.Create.restype = ctypes.c_int

sgfplib.OpenDevice.argtypes = [ctypes.c_int]
sgfplib.OpenDevice.restype = ctypes.c_int

sgfplib.GetImageSize.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
sgfplib.GetImageSize.restype = ctypes.c_int

sgfplib.GetImage.argtypes = [ctypes.POINTER(ctypes.c_ubyte)]
sgfplib.GetImage.restype = ctypes.c_int

sgfplib.CloseDevice.argtypes = []
sgfplib.CloseDevice.restype = ctypes.c_int

sgfplib.Destroy.argtypes = []
sgfplib.Destroy.restype = ctypes.c_int

# Initialize SDK
ret = sgfplib.Create()
if ret != 0:
    print("Failed to initialize SecuGen library.")
    exit()
print("SecuGen SDK initialized successfully.")

# Open the fingerprint scanner device
ret = sgfplib.OpenDevice(0)
if ret != 0:
    print("Failed to open device.")
    exit()
print("Device opened successfully.")

# Get image size
width = ctypes.c_int()
height = ctypes.c_int()
ret = sgfplib.GetImageSize(ctypes.byref(width), ctypes.byref(height))
if ret != 0:
    print("Failed to get image size.")
    exit()
print(f"Image size: {width.value}x{height.value}")

# Capture fingerprint image
image_buffer = (ctypes.c_ubyte * (width.value * height.value))()
ret = sgfplib.GetImage(image_buffer)
if ret != 0:
    print("Failed to capture fingerprint image.")
    exit()
print("Fingerprint image captured successfully.")

# Convert to NumPy array and reshape
fingerprint_image = np.frombuffer(image_buffer, dtype=np.uint8).reshape((height.value, width.value))

# Display the fingerprint image
cv2.imshow("Fingerprint Image", fingerprint_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Close device
sgfplib.CloseDevice()
sgfplib.Destroy()
print("Device closed and SDK destroyed.")
