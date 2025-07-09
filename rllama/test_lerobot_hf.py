"""
Test LeRobot with Python 3.12
"""
import sys
import importlib
import inspect

print(f"Python version: {sys.version}")

try:
    import lerobot
    print(f"Successfully imported LeRobot from {lerobot.__file__}")
    print(f"LeRobot package contents:")
    
    # List all non-private attributes
    for name in dir(lerobot):
        if not name.startswith("__"):
            try:
                attr = getattr(lerobot, name)
                print(f"- {name} ({type(attr).__name__})")
            except Exception as e:
                print(f"- {name} (Error: {e})")
    
    # Try to find a specific LeRobot class or function that might work
    print("\nAttempting to find LeRobot functionality:")
    
    # Look for a Robot class
    if hasattr(lerobot, "Robot"):
        print("Found lerobot.Robot class")
        robot_class = lerobot.Robot
        print(f"Methods: {[m for m in dir(robot_class) if not m.startswith('__')]}")
        
        # Try to create a robot instance
        try:
            robot = robot_class()
            print(f"Successfully created robot instance")
            print(f"Robot attributes: {[a for a in dir(robot) if not a.startswith('__')]}")
        except Exception as e:
            print(f"Could not create robot instance: {e}")
    
    # Look for a control function
    if hasattr(lerobot, "control"):
        print("\nFound lerobot.control function")
        control_fn = lerobot.control
        print(f"Signature: {inspect.signature(control_fn)}")
    
    # Look for any simulation-related functions
    sim_functions = [name for name in dir(lerobot) if "sim" in name.lower() and callable(getattr(lerobot, name))]
    if sim_functions:
        print(f"\nFound simulation-related functions: {sim_functions}")

except ImportError as e:
    print(f"Error importing LeRobot: {e}")