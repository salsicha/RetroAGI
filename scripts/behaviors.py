
import py_trees
import yaml
import random

class CommandBehavior(py_trees.behaviour.Behaviour):
    def __init__(self, name, key):
        super().__init__(name)
        self.key = key
        self.selected = False

    def update(self):
        self.logger.debug(f"Evaluating command: {self.name}")
        # Simulate decision-making logic (e.g., game state)
        self.selected = random.choice([True, False])
        if self.selected:
            self.logger.debug(f"Command selected: {self.key}")
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE

class MarioController:
    def __init__(self, config_path):
        self.config_path = config_path
        self.tree = None
        self.commands = []
        self._load_config()
        self._build_tree()

    def _load_config(self):
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
            self.commands = sorted(config['commands'], key=lambda x: x['priority'])

    def _build_tree(self):
        root = py_trees.composites.Selector("MarioCommandSelector", memory=False)
        for cmd in self.commands:
            behavior = CommandBehavior(cmd['name'], cmd['key'])
            root.add_child(behavior)
        self.tree = py_trees.trees.BehaviourTree(root)

    def tick(self):
        self.tree.tick()
        for node in self.tree.root.iterate():
            if isinstance(node, CommandBehavior) and node.status == py_trees.common.Status.SUCCESS:
                print(f"Executing command: {node.key}")
                return node.key
        print("No command selected")
        return None


controller = MarioController("behavior_config.yaml")

# Simulate game loop
for _ in range(10):
    controller.tick()