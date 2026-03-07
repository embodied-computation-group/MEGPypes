from pathlib import Path
from dataclasses import dataclass
import re
from bids import BIDSLayout

@dataclass
class Run:
    subject: str
    path: str
    session: str | None = None
    task: str | None = None

class RunFinder:
    def __init__(self, root):
        self.root = Path(root)

    def find_runs(self, 
                  subject_list: list[str] | None = None,
                  session_list: str | list[str] | None = None,
                  task_list: str | list[str] | None = None):
        
        all_runs = self._discover_runs()

        # Filter runs based on provided criteria
        filtered_runs = []
        for run in all_runs:
            if subject_list and run.subject not in subject_list:
                continue
            if session_list and run.session not in session_list:
                continue
            if task_list and run.task not in task_list:
                continue
            filtered_runs.append(run)

        return filtered_runs

    def _discover_runs(self):

        runs = []

        # BIDS case
        if (self.root / "dataset_description.json").exists():

            layout = BIDSLayout(self.root)

            for f in layout.get(extension=[".fif", ".ds"], return_type="filename"):
                ent = layout.parse_file_entities(f)

                runs.append(
                    Run(
                        subject=ent.get("subject"),
                        path=f,
                        session=ent.get("session"),
                        task=ent.get("task")
                    )
                )

            return runs

        # flat CTF dataset
        pattern = re.compile(r"(?P<sub>\d+)_.*\.ds")

        for ds in self.root.glob("*.ds"):

            m = pattern.match(ds.name)
            if not m:
                continue

            runs.append(
                Run(
                    subject=m.group("sub"),
                    path=str(ds)
                )
            )

        return runs