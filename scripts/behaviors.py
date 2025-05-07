import yaml
import py_trees
from pynput.keyboard import Key, Controller

# Load configuration from YAML file
with open('behavior_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract flags
up_flag = config.get('up', False)
down_flag = config.get('down', False)
left_flag = config.get('left', False)
right_flag = config.get('right', False)

# Initialize keyboard controller
keyboard = Controller()

# Define condition classes
class UpCondition(py_trees.behaviour.Behaviour):
    def __init__(self):
        super(UpCondition, self).__init__("UpCondition")
        self.up = up_flag

    def update(self):
        if self.up:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE

class DownCondition(py_trees.behaviour.Behaviour):
    def __init__(self):
        super(DownCondition, self).__init__("DownCondition")
        self.down = down_flag

    def update(self):
        if self.down:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE

class LeftCondition(py_trees.behaviour.Behaviour):
    def __init__(self):
        super(LeftCondition, self).__init__("LeftCondition")
        self.left = left_flag

    def update(self):
        if self.left:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE

class RightCondition(py_trees.behaviour.Behaviour):
    def __init__(self):
        super(RightCondition, self).__init__("RightCondition")
        self.right = right_flag

    def update(self):
        if self.right:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE

# Define action classes
class SendUpAction(py_trees.behaviour.Behaviour):
    def __init__(self):
        super(SendUpAction, self).__init__("SendUpAction")

    def setup(self):
        return True

    def update(self):
        keyboard.press(Key.up)
        keyboard.release(Key.up)
        return py_trees.common.Status.SUCCESS

class SendDownAction(py_trees.behaviour.Behaviour):
    def __init__(self):
        super(SendDownAction, self).__init__("SendDownAction")

    def setup(self):
        return True

    def update(self):
        keyboard.press(Key.down)
        keyboard.release(Key.down)
        return py_trees.common.Status.SUCCESS

class SendLeftAction(py_trees.behaviour.Behaviour):
    def __init__(self):
        super(SendLeftAction, self).__init__("SendLeftAction")

    def setup(self):
        return True

    def update(self):
        keyboard.press(Key.left)
        keyboard.release(Key.left)
        return py_trees.common.Status.SUCCESS

class SendRightAction(py_trees.behaviour.Behaviour):
    def __init__(self):
        super(SendRightAction, self).__init__("SendRightAction")

    def setup(self):
        return True

    def update(self):
        keyboard.press(Key.right)
        keyboard.release(Key.right)
        return py_trees.common.Status.SUCCESS

# Create sequences for each condition-action pair
up_sequence = py_trees.composites.Sequence("UpSequence", memory=True)
up_sequence.add_child(UpCondition())
up_sequence.add_child(SendUpAction())

down_sequence = py_trees.composites.Sequence("DownSequence", memory=True)
down_sequence.add_child(DownCondition())
down_sequence.add_child(SendDownAction())

left_sequence = py_trees.composites.Sequence("LeftSequence", memory=True)
left_sequence.add_child(LeftCondition())
left_sequence.add_child(SendLeftAction())

right_sequence = py_trees.composites.Sequence("RightSequence", 
memory=True)
right_sequence.add_child(RightCondition())
right_sequence.add_child(SendRightAction())

# Create parallel node with policy to run all children
root = py_trees.composites.Parallel(
    name="RootParallel",
    policy=py_trees.common.ParallelPolicy.SUCCESSOR  # Or any other policy
)

root.add_child(up_sequence)
root.add_child(down_sequence)
root.add_child(left_sequence)
root.add_child(right_sequence)

# Run the behavior tree
root.tick()
