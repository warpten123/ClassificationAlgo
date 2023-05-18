from pathlib import Path
from typing import Iterable, Any

from pdfminer.high_level import extract_pages


class Testing():
    def show_ltitem_hierarchy(self, o: Any, depth=0):
        """Show location and text of LTItem and all its descendants"""
        if depth == 0:
            print('element                        x1  y1  x2  y2   text')
            print('------------------------------ --- --- --- ---- -----')

        print(
            # f'{self.get_indented_name(o, depth):<30.30s} '
            # f'{self.get_optional_bbox(o)} '
            f'{self.get_optional_text(o)}'
        )

        if isinstance(o, Iterable):
            for i in o:
                self.show_ltitem_hierarchy(i, depth=depth + 1)

    def get_indented_name(self, o: Any, depth: int) -> str:
        """Indented name of LTItem"""
        return '  ' * depth + o.__class__.__name__

    def get_optional_bbox(self, o: Any) -> str:
        """Bounding box of LTItem if available, otherwise empty string"""
        if hasattr(o, 'bbox'):
            return ''.join(f'{i:<4.0f}' for i in o.bbox)
        return ''

    def get_optional_text(self, o: Any) -> str:
        """Text of LTItem if available, otherwise empty string"""
        if hasattr(o, 'get_text'):
            return o.get_text().strip()
        return ''

    def main(self):
        path = Path(
            'C:\\Users\\Dennis\\Documents\\COMICS\\College\\Test PDF\\Test Set\\testing.pdf').expanduser()
        pages = extract_pages(path, 0)
        print(pages)
        self.show_ltitem_hierarchy(pages)
        return "shit"
