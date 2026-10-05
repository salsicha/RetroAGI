"""The documentation only points at files, modules and commands that exist."""

import re
import shlex
import unittest
from pathlib import Path

DOCS = [Path("README.md"), Path("scripts/vit/README.md"), *sorted(Path("docs").glob("*.md"))]
# Documents kept from before the layered agent; they describe removed code
# and link to removed documents, so they are not checked.
EARLIER_NOTES = {
    "ai-teaching-curriculum.md",
    "hierarchical-self-supervised-planning.md",
    "issues.md",
    "universal-embodied-framework.md",
    "universal-retro-oracle.md",
}
CURRENT = [path for path in DOCS if path.name not in EARLIER_NOTES]


def code_lines(text):
    """Lines of the documents' bash blocks, with continuation lines joined."""
    for block in re.findall(r"```bash\n(.*?)```", text, re.S):
        yield from block.replace("\\\n", " ").splitlines()


class TestDocumentation(unittest.TestCase):
    def test_links_point_at_existing_files(self):
        for doc in CURRENT:
            for target in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", doc.read_text(encoding="utf-8")):
                if target.startswith("http"):
                    continue
                with self.subTest(doc=str(doc), target=target):
                    self.assertTrue((doc.parent / target).exists())

    def test_commands_name_existing_scripts_modules_and_subcommands(self):
        from retroagi.stages.block_smb.cli import build_parser

        subcommands = build_parser()._subparsers._group_actions[0].choices
        for doc in CURRENT:
            for line in code_lines(doc.read_text(encoding="utf-8")):
                words = shlex.split(line) if line.strip() else []
                if not words:
                    continue
                with self.subTest(doc=str(doc), line=line):
                    if words[0] == "retroagi-block-smb":
                        self.assertIn(words[1], subcommands)
                    elif words[:2] == ["python", "-m"] and words[2].startswith("retroagi"):
                        module = Path(*words[2].split("."))
                        self.assertTrue(
                            module.with_suffix(".py").exists() or (module / "__init__.py").exists()
                        )
                    elif words[0] == "python" and words[1].endswith(".py"):
                        self.assertTrue(Path(words[1]).exists())

    def test_current_docs_name_no_removed_commands(self):
        removed = ("retroagi train", "retroagi evaluate", "retroagi check-env", "retroagi promote")
        for doc in CURRENT:
            text = doc.read_text(encoding="utf-8")
            for command in removed:
                with self.subTest(doc=str(doc), command=command):
                    self.assertNotIn(command, text)


if __name__ == "__main__":
    unittest.main()
