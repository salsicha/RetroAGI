"""Screen capture utility for capturing game screens."""
import mss
import numpy as np
import cv2


class ScreenCapture:
    """A class for capturing the screen."""

    def __init__(self, monitor_number=1):
        self.sct = mss.mss()
        self.monitor_number = monitor_number

    def capture_screen(self):
        """
        Capture the screen and return it as a numpy array.
        """
        sct_img = self.sct.grab(self.sct.monitors[self.monitor_number])
        img = np.array(sct_img)
        return img

    def capture_window(self, window_name):
        """
        Capture a specific window and return it as a numpy array.
        """
        # This is a placeholder for capturing a specific window.
        # Implementation depends on the OS and installed libraries.
        # For now, it captures the whole screen.
        print(f"Warning: Capturing the whole screen, not just the window '{window_name}'.")
        return self.capture_screen()