import argparse
import re
import shutil
from pathlib import Path

import nbconvert
from nbconvert.preprocessors import Preprocessor


class DatasigPreprocessor(Preprocessor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title_seen = False

    def _generate_lines(self, source, resources):
        if source.startswith("<div>"):
            return

        metadata = resources["metadata"]
        if "referenced_images" not in resources:
            resources["referenced_images"] = {}


        for line in source.splitlines():
            line = line.strip()
            if (re.match(r"#+ https://(.+)", line)) is not None:
                continue

            if line.startswith("# "):
                if self.title_seen:
                    yield "#" + line
                else:
                    yield line
                    self.title_seen = True
                continue

            if (m := re.match(r"^#{3,}", line)) is not None:
                yield "###" + line[m.end():]
                continue

            for m in re.finditer(r"""!\[[^]]+\]\((([^)\s]+)(\s+"\w+")?)\)""",
                                 line):
                p = Path(metadata["path"], m.group(2))
                begin, end = m.span(1)
                filename = Path(m.group(2)).name
                new_path = Path(f"{metadata['name']}_{filename}")

                resources["referenced_images"][str(new_path)] = p

                new_line = line[:begin] + str(new_path) + line[end:]
                line = new_line

            yield line

    def _process_md(self, md_cell, resources):
        md_cell["source"] = "\n".join(
            self._generate_lines(md_cell["source"], resources)
        )
        return md_cell

    def preprocess_cell(self, cell, resources, index):
        if cell["cell_type"] == "raw":
            if cell["source"].contains("<div>"):
                return None, resources
        if cell["cell_type"] == "markdown":
            self._process_md(cell, resources)
        return cell, resources


def get_exporter():
    from traitlets.config import Config

    config = Config()
    config.RSTExporter.preprocessors = [DatasigPreprocessor]
    return nbconvert.exporters.RSTExporter(config=config)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--output-dir", type=Path, default=Path.cwd())

    parser.add_argument("notebooks", type=Path, nargs="+")

    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(exist_ok=True, parents=True)

    for p in args.notebooks:
        exporter = get_exporter()

        resources = nbconvert.exporters.ResourcesDict()
        resources["unique_key"] = p.stem

        out_path = Path(args.output_dir, p.stem).with_suffix(".rst")
        (nb, r) = exporter.from_filename(str(p), resources=resources)
        out_path.write_text(nb)

        for out_name, data in r["outputs"].items():
            op = output_dir / out_name
            op.write_bytes(data)

        for out_path, in_path in r["referenced_images"].items():
            op = output_dir / out_path
            shutil.copy(in_path, op)


if __name__ == "__main__":
    main()
