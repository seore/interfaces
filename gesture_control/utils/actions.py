import sys
import time

try:
    import pyautogui
    pyautogui.FAILSAFE = False
except Exception:
    pyautogui = None

def _is_macos():
    return sys.platform == "darwin"

def _is_windows():
    return sys.platform.startswith("win")

def do_action(action: str):
    """
    Map gestures -> browser controls.
    Works on Windows + macOS. (Linux also mostly works.)
    """
    if pyautogui is None:
        print("pyautogui not available. Install it or run without actions.")
        return

    # Common actions
    if action == "click":
        pyautogui.click()
        return

    if action == "scroll_up":
        pyautogui.scroll(200)
        return

    if action == "scroll_down":
        pyautogui.scroll(-200)
        return

    # Browser back/forward hotkeys differ:
    # Windows Chrome/Edge: Alt+Left / Alt+Right
    # macOS Chrome/Safari: Command+[ / Command+]
    if action == "browser_back":
        if _is_macos():
            pyautogui.hotkey("command", "[")
        else:
            pyautogui.hotkey("alt", "left")
        return

    if action == "browser_forward":
        if _is_macos():
            pyautogui.hotkey("command", "]")
        else:
            pyautogui.hotkey("alt", "right")
        return

    if action == "new_tab":
        if _is_macos():
            pyautogui.hotkey("command", "t")
        else:
            pyautogui.hotkey("ctrl", "t")
        return

def gesture_to_action(gesture: str):
    if gesture == "pinch":
        return "click"
    if gesture == "thumbs_up":
        return "scroll_up"
    if gesture == "fist":
        return "scroll_down"
    if gesture == "swipe_left":
        return "browser_back"
    if gesture == "swipe_right":
        return "browser_forward"
    return None
