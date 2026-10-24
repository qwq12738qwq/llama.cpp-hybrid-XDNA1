import pyxrt
import time

print("pyxrt imported")

try:
    print("creating device...")
    device = pyxrt.device(0)
    print(f"device created: {device}")
except Exception as e:
    print(f"device error: {e}")

try:
    print("listing devices...")
    devs = pyxrt.device.get_devices()
    print(f"devices: {devs}")
except Exception as e:
    print(f"list error: {e}")
